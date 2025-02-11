import os
import glob
import io
import argparse
import multiprocessing
from typing import List, Tuple

import numpy as np
import pandas as pd
import imageio.v3 as iio
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms

class SimpleVideoProcessor:
    def __init__(self, size=(80, 80), mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225)):
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def __call__(self, frames, return_tensors=None):
        processed_frames = [self.transform(frame) for frame in frames]
        pixel_values = torch.stack(processed_frames)
        if return_tensors == "pt":
            return {"pixel_values": pixel_values}
        else:
            return pixel_values

def load_volumes(bbox_name: str, raw_base_dir: str, seg_base_dir: str, add_mask_base_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw_dir = os.path.join(raw_base_dir, bbox_name)
    seg_dir = os.path.join(seg_base_dir, bbox_name)

    if bbox_name.startswith("bbox"):
        bbox_num = bbox_name.replace("bbox", "")
        add_mask_dir = os.path.join(add_mask_base_dir, f"bbox_{bbox_num}")
    else:
        add_mask_dir = os.path.join(add_mask_base_dir, bbox_name)

    raw_tif_files = sorted(glob.glob(os.path.join(raw_dir, 'slice_*.tif')))
    seg_tif_files = sorted(glob.glob(os.path.join(seg_dir, 'slice_*.tif')))
    add_mask_tif_files = sorted(glob.glob(os.path.join(add_mask_dir, 'slice_*.tif')))

    if not (len(raw_tif_files) == len(seg_tif_files) == len(add_mask_tif_files)):
        return None, None, None

    try:
        raw_vol = np.stack([iio.imread(f) for f in raw_tif_files], axis=0)
        seg_vol = np.stack([iio.imread(f).astype(np.uint32) for f in seg_tif_files], axis=0)
        add_mask_vol = np.stack([iio.imread(f).astype(np.uint32) for f in add_mask_tif_files], axis=0)
        return raw_vol, seg_vol, add_mask_vol
    except Exception as e:
        return None, None, None

def create_segmented_cube(
    raw_vol: np.ndarray,
    seg_vol: np.ndarray,
    add_mask_vol: np.ndarray,
    central_coord: Tuple[int, int, int],
    side1_coord: Tuple[int, int, int],
    side2_coord: Tuple[int, int, int],
    segmentation_type: int,
    subvolume_size: int = 80,
    alpha: float = 0.3,
    input_mask: bool = False  # New parameter
) -> np.ndarray:
    def create_segment_masks(segmentation_volume, s1_coord, s2_coord):
        x1, y1, z1 = s1_coord
        x2, y2, z2 = s2_coord
        seg_id_1 = segmentation_volume[z1, y1, x1]
        seg_id_2 = segmentation_volume[z2, y2, x2]

        mask_1 = (segmentation_volume == seg_id_1) if seg_id_1 != 0 else np.zeros_like(segmentation_volume, dtype=bool)
        mask_2 = (segmentation_volume == seg_id_2) if seg_id_2 != 0 else np.zeros_like(segmentation_volume, dtype=bool)
        return mask_1, mask_2

    mask_1_full, mask_2_full = create_segment_masks(seg_vol, side1_coord, side2_coord)
    mask_3_full = (add_mask_vol > 0)

    half_size = subvolume_size // 2
    cx, cy, cz = central_coord
    x_start, x_end = max(cx - half_size, 0), min(cx + half_size, raw_vol.shape[2])
    y_start, y_end = max(cy - half_size, 0), min(cy + half_size, raw_vol.shape[1])
    z_start, z_end = max(cz - half_size, 0), min(cz + half_size, raw_vol.shape[0])

    sub_raw = raw_vol[z_start:z_end, y_start:y_end, x_start:x_end]
    sub_mask_1 = mask_1_full[z_start:z_end, y_start:y_end, x_start:x_end]
    sub_mask_2 = mask_2_full[z_start:z_end, y_start:y_end, x_start:x_end]
    sub_mask_3 = mask_3_full[z_start:z_end, y_start:y_end, x_start:x_end]

    pad_z = subvolume_size - sub_raw.shape[0]
    pad_y = subvolume_size - sub_raw.shape[1]
    pad_x = subvolume_size - sub_raw.shape[2]

    if pad_z > 0 or pad_y > 0 or pad_x > 0:
        sub_raw = np.pad(sub_raw, ((0, pad_z), (0, pad_y), (0, pad_x)), mode='constant', constant_values=0)
        sub_mask_1 = np.pad(sub_mask_1, ((0, pad_z), (0, pad_y), (0, pad_x)), mode='constant', constant_values=False)
        sub_mask_2 = np.pad(sub_mask_2, ((0, pad_z), (0, pad_y), (0, pad_x)), mode='constant', constant_values=False)
        sub_mask_3 = np.pad(sub_mask_3, ((0, pad_z), (0, pad_y), (0, pad_x)), mode='constant', constant_values=False)

    sub_raw = sub_raw[:subvolume_size, :subvolume_size, :subvolume_size]
    sub_mask_1 = sub_mask_1[:subvolume_size, :subvolume_size, :subvolume_size]
    sub_mask_2 = sub_mask_2[:subvolume_size, :subvolume_size, :subvolume_size]
    sub_mask_3 = sub_mask_3[:subvolume_size, :subvolume_size, :subvolume_size]

    overlaid_cube = np.zeros((subvolume_size, subvolume_size, 3, subvolume_size), dtype=np.uint8)

    for z in range(subvolume_size):
        raw_slice = sub_raw[z].astype(np.float32)
        mn, mx = raw_slice.min(), raw_slice.max()
        if mx > mn:
            raw_slice = (raw_slice - mn) / (mx - mn)
        else:
            raw_slice = raw_slice - mn

        if input_mask:  # New masking logic
            combined_mask = np.zeros_like(raw_slice, dtype=np.float32)
            if segmentation_type in [1, 3, 5]:
                combined_mask = np.logical_or(combined_mask, sub_mask_1[z])
            if segmentation_type in [2, 3, 5]:
                combined_mask = np.logical_or(combined_mask, sub_mask_2[z])
            if segmentation_type in [4, 5]:
                combined_mask = np.logical_or(combined_mask, sub_mask_3[z])
            
            masked_raw = raw_slice * combined_mask
            masked_rgb = np.stack([masked_raw]*3, axis=-1)
            overlaid_image = (masked_rgb * 255).astype(np.uint8)
        else:  # Original overlay logic
            raw_rgb = np.stack([raw_slice]*3, axis=-1)
            mask1_rgb = np.zeros_like(raw_rgb)
            mask2_rgb = np.zeros_like(raw_rgb)
            mask3_rgb = np.zeros_like(raw_rgb)

            if segmentation_type in [1, 3, 5]:
                mask1_rgb[sub_mask_1[z]] = [1, 0, 0]
            if segmentation_type in [2, 3, 5]:
                mask2_rgb[sub_mask_2[z]] = [0, 0, 1]
            if segmentation_type in [4, 5]:
                mask3_rgb[sub_mask_3[z]] = [0, 1, 0]

            combined_masks = mask1_rgb + mask2_rgb + mask3_rgb
            combined_masks = np.clip(combined_masks, 0, 1)
            overlaid_image = (1 - alpha) * raw_rgb + alpha * combined_masks
            overlaid_image = (np.clip(overlaid_image, 0, 1) * 255).astype(np.uint8)

        overlaid_cube[:, :, :, z] = overlaid_image

    return overlaid_cube

class VideoMAEDataset(Dataset):
    def __init__(self, vol_data_dict: dict, synapse_df: pd.DataFrame, processor,
                 segmentation_type: int, subvol_size: int = 80, num_frames: int = 16,
                 alpha: float = 0.3, input_mask: bool = False):  # Added input_mask
        self.vol_data_dict = vol_data_dict
        self.synapse_df = synapse_df.reset_index(drop=True)
        self.processor = processor
        self.segmentation_type = segmentation_type
        self.subvol_size = subvol_size
        self.num_frames = num_frames
        self.alpha = alpha
        self.input_mask = input_mask  # Store input_mask flag

    def __len__(self):
        return len(self.synapse_df)

    def __getitem__(self, idx):
        syn_info = self.synapse_df.iloc[idx]
        bbox_name = syn_info['bbox_name']
        raw_vol, seg_vol, add_mask_vol = self.vol_data_dict.get(bbox_name, (None, None, None))

        if raw_vol is None:
            return torch.zeros((self.num_frames, 3, self.subvol_size, self.subvol_size), dtype=torch.float32), syn_info, bbox_name

        central_coord = (int(syn_info['central_coord_1']), int(syn_info['central_coord_2']), int(syn_info['central_coord_3']))
        side1_coord = (int(syn_info['side_1_coord_1']), int(syn_info['side_1_coord_2']), int(syn_info['side_1_coord_3']))
        side2_coord = (int(syn_info['side_2_coord_1']), int(syn_info['side_2_coord_2']), int(syn_info['side_2_coord_3']))

        overlaid_cube = create_segmented_cube(
            raw_vol=raw_vol,
            seg_vol=seg_vol,
            add_mask_vol=add_mask_vol,
            central_coord=central_coord,
            side1_coord=side1_coord,
            side2_coord=side2_coord,
            segmentation_type=self.segmentation_type,
            subvolume_size=self.subvol_size,
            alpha=self.alpha,
            input_mask=self.input_mask  # Pass the flag
        )

        frames = [overlaid_cube[..., z] for z in range(overlaid_cube.shape[3])]
        if len(frames) < self.num_frames:
            frames += [frames[-1]] * (self.num_frames - len(frames))
        elif len(frames) > self.num_frames:
            indices = np.linspace(0, len(frames)-1, self.num_frames, dtype=int)
            frames = [frames[i] for i in indices]

        inputs = self.processor(frames, return_tensors="pt")
        return inputs["pixel_values"].squeeze(0).float(), syn_info, bbox_name

def parse_args():
    parser = argparse.ArgumentParser(description="VideoMAE Pre-training Script with Segmented Videos and Additional Masks")
    parser.add_argument('--raw_base_dir', type=str, default='raw')
    parser.add_argument('--seg_base_dir', type=str, default='seg')
    parser.add_argument('--add_mask_base_dir', type=str, default='')
    parser.add_argument('--bbox_name', type=str, default=['bbox1'], nargs='+')
    parser.add_argument('--excel_file', type=str, default='')
    parser.add_argument('--csv_output_dir', type=str, default='csv_outputs')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--size', type=tuple, default=(80,80))
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--subvol_size', type=int, default=80)
    parser.add_argument('--num_frames', type=int, default=80)
    parser.add_argument('--mask_ratio', type=float, default=0.75)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--resume_checkpoint', type=str, default=None)
    parser.add_argument('--save_gifs_dir', type=str, default='gifs')
    parser.add_argument('--num_gifs', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=0.3)
    parser.add_argument('--segmentation_type', type=int, default=5, choices=range(0, 6))
    parser.add_argument('--input_mask', action='store_true', # Added argument
                       help='Mask input image using segmentation_type regions')
    args, _ = parser.parse_known_args()

    return args

def main(args):
    processor = SimpleVideoProcessor(size=(80, 80))
    vol_data_dict = {}
    
    for bbox_name in args.bbox_name:
        raw_vol, seg_vol, add_mask_vol = load_volumes(
            bbox_name=bbox_name,
            raw_base_dir=args.raw_base_dir,
            seg_base_dir=args.seg_base_dir,
            add_mask_base_dir=args.add_mask_base_dir
        )
        if raw_vol is not None:
            vol_data_dict[bbox_name] = (raw_vol, seg_vol, add_mask_vol)

    synapse_dfs = []
    for bbox_name in args.bbox_name:
        excel_path = os.path.join(args.excel_file, f"{bbox_name}.xlsx")
        if os.path.exists(excel_path):
            df = pd.read_excel(excel_path)
            df['bbox_name'] = bbox_name
            synapse_dfs.append(df)
            
    syn_df = pd.concat(synapse_dfs, ignore_index=True)
    dataset = VideoMAEDataset(
        vol_data_dict=vol_data_dict,
        synapse_df=syn_df,
        processor=processor,
        segmentation_type=args.segmentation_type,
        subvol_size=args.subvol_size,
        num_frames=args.num_frames,
        alpha=args.alpha,
        input_mask=args.input_mask  # Pass the flag
    )

    cubes = []
    syn_info_list = []
    for idx in range(len(dataset)):
        pixel_values, syn_info, _ = dataset[idx]
        cubes.append(pixel_values)
        syn_info_list.append(syn_info)

    print(f"Processed {len(cubes)} cubes successfully.")
    return cubes, pd.DataFrame(syn_info_list)

if __name__ == "__main__":
    args = parse_args()
    cubes, sys_inf = main(args)
    print(f"Final output: {len(cubes)} cubes")
