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

# import wandb  # Uncomment if using Weights & Biases for logging


class SimpleVideoProcessor:
    def __init__(self, size=(80, 80), mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225)):
        """
        Initializes the processor with resizing and normalization transforms.

        Args:
            size (tuple): Desired output size (height, width).
            mean (tuple): Mean for normalization.
            std (tuple): Standard deviation for normalization.
        """
        self.transform = transforms.Compose([
            transforms.ToPILImage(),          # Convert NumPy array to PIL Image
            transforms.Resize(size),          # Resize to desired size
            transforms.ToTensor(),            # Convert PIL Image to Tensor
            transforms.Normalize(mean=mean, std=std),  # Normalize
        ])

    def __call__(self, frames, return_tensors=None):
        """
        Processes a list of frames.

        Args:
            frames (List[np.ndarray]): List of frames as NumPy arrays.
            return_tensors (str, optional): Type of tensors to return. Defaults to None.

        Returns:
            dict or torch.Tensor: Dictionary containing processed pixel values or tensor.
        """
        # Apply transformations to each frame
        processed_frames = [self.transform(frame) for frame in frames]

        # Stack frames to create a tensor of shape (num_frames, 3, H, W)
        pixel_values = torch.stack(processed_frames)

        if return_tensors == "pt":
            return {"pixel_values": pixel_values}
        else:
            return pixel_values


def load_volumes(bbox_name: str, raw_base_dir: str, seg_base_dir: str, add_mask_base_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load raw volume, segmentation volume, and additional mask volume for a bounding box.

    Args:
        bbox_name (str): Name of the bounding box directory (e.g., 'bbox1').
        raw_base_dir (str): Base directory for raw data.
        seg_base_dir (str): Base directory for segmentation data.
        add_mask_base_dir (str): Base directory for additional masks.

    Returns:
        tuple: (raw_vol, seg_vol, add_mask_vol) each as np.ndarray
    """
    raw_dir = os.path.join(raw_base_dir, bbox_name)
    seg_dir = os.path.join(seg_base_dir, bbox_name)

    # Transform 'bbox1' to 'bbox_1' for additional masks
    if bbox_name.startswith("bbox"):
        bbox_num = bbox_name.replace("bbox", "")
        add_mask_dir = os.path.join(add_mask_base_dir, f"bbox_{bbox_num}")
    else:
        add_mask_dir = os.path.join(add_mask_base_dir, bbox_name)

    raw_tif_files = sorted(glob.glob(os.path.join(raw_dir, 'slice_*.tif')))
    seg_tif_files = sorted(glob.glob(os.path.join(seg_dir, 'slice_*.tif')))
    add_mask_tif_files = sorted(glob.glob(os.path.join(add_mask_dir, 'slice_*.tif')))

    if len(raw_tif_files) == 0:
        print(f"No raw files found for {bbox_name} in {raw_dir}")
        return None, None, None

    if len(seg_tif_files) == 0:
        print(f"No segmentation files found for {bbox_name} in {seg_dir}")
        return None, None, None

    if len(add_mask_tif_files) == 0:
        print(f"No additional mask files found for {bbox_name} in {add_mask_dir}")
        return None, None, None

    if not (len(raw_tif_files) == len(seg_tif_files) == len(add_mask_tif_files)):
        print(f"Mismatch in number of raw, seg, and additional mask slices for {bbox_name}. Skipping.")
        return None, None, None

    try:
        raw_vol = np.stack([iio.imread(f) for f in raw_tif_files], axis=0)  # shape: (Z, Y, X)
        seg_vol = np.stack([iio.imread(f).astype(np.uint32) for f in seg_tif_files], axis=0)
        add_mask_vol = np.stack([iio.imread(f).astype(np.uint32) for f in add_mask_tif_files], axis=0)
        return raw_vol, seg_vol, add_mask_vol
    except Exception as e:
        print(f"Error loading volumes for {bbox_name}: {e}")
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
    alpha: float = 0.3
) -> np.ndarray:
    """
    Constructs an 80x80x80 segmented 3D cube around the specified synapse coordinates
    and overlays selected segmentation masks on the raw data with specified transparency for each slice.

    Args:
        raw_vol (np.ndarray): Raw volumetric data.
        seg_vol (np.ndarray): Segmentation volumetric data.
        add_mask_vol (np.ndarray): Additional mask volumetric data.
        central_coord (tuple): Central coordinate (x, y, z).
        side1_coord (tuple): Side 1 coordinate (x, y, z).
        side2_coord (tuple): Side 2 coordinate (x, y, z).
        segmentation_type (int): Type of segmentation overlay (0-5).
        subvolume_size (int, optional): Size of the subvolume. Defaults to 80.
        alpha (float, optional): Transparency factor. Defaults to 0.3.

    Returns:
        np.ndarray: Overlaid cube of shape (height, width, 3, depth).
    """

    def create_segment_masks(segmentation_volume, s1_coord, s2_coord):
        x1, y1, z1 = s1_coord
        x2, y2, z2 = s2_coord
        # Validate within volume
        if not (0 <= z1 < segmentation_volume.shape[0] and
                0 <= y1 < segmentation_volume.shape[1] and
                0 <= x1 < segmentation_volume.shape[2]):
            raise ValueError("Side1 coordinates are out of bounds.")

        if not (0 <= z2 < segmentation_volume.shape[0] and
                0 <= y2 < segmentation_volume.shape[1] and
                0 <= x2 < segmentation_volume.shape[2]):
            raise ValueError("Side2 coordinates are out of bounds.")

        seg_id_1 = segmentation_volume[z1, y1, x1]
        seg_id_2 = segmentation_volume[z2, y2, x2]

        # If seg_id == 0, it means no segment at that voxel
        if seg_id_1 == 0:
            mask_1 = np.zeros_like(segmentation_volume, dtype=bool)
        else:
            mask_1 = (segmentation_volume == seg_id_1)

        if seg_id_2 == 0:
            mask_2 = np.zeros_like(segmentation_volume, dtype=bool)
        else:
            mask_2 = (segmentation_volume == seg_id_2)

        return mask_1, mask_2

    # Build masks
    mask_1_full, mask_2_full = create_segment_masks(seg_vol, side1_coord, side2_coord)
    mask_3_full = (add_mask_vol > 0)  # Assuming binary masks; adjust if necessary

    # Define subvolume bounds
    half_size = subvolume_size // 2
    cx, cy, cz = central_coord

    x_start, x_end = max(cx - half_size, 0), min(cx + half_size, raw_vol.shape[2])
    y_start, y_end = max(cy - half_size, 0), min(cy + half_size, raw_vol.shape[1])
    z_start, z_end = max(cz - half_size, 0), min(cz + half_size, raw_vol.shape[0])

    # Extract subvolumes
    sub_raw = raw_vol[z_start:z_end, y_start:y_end, x_start:x_end]
    sub_mask_1 = mask_1_full[z_start:z_end, y_start:y_end, x_start:x_end]
    sub_mask_2 = mask_2_full[z_start:z_end, y_start:y_end, x_start:x_end]
    sub_mask_3 = mask_3_full[z_start:z_end, y_start:y_end, x_start:x_end]

    # Pad if smaller than subvolume_size
    pad_z = subvolume_size - sub_raw.shape[0]
    pad_y = subvolume_size - sub_raw.shape[1]
    pad_x = subvolume_size - sub_raw.shape[2]

    if pad_z > 0 or pad_y > 0 or pad_x > 0:
        sub_raw = np.pad(sub_raw, ((0, pad_z), (0, pad_y), (0, pad_x)),
                         mode='constant', constant_values=0)
        sub_mask_1 = np.pad(sub_mask_1, ((0, pad_z), (0, pad_y), (0, pad_x)),
                            mode='constant', constant_values=False)
        sub_mask_2 = np.pad(sub_mask_2, ((0, pad_z), (0, pad_y), (0, pad_x)),
                            mode='constant', constant_values=False)
        sub_mask_3 = np.pad(sub_mask_3, ((0, pad_z), (0, pad_y), (0, pad_x)),
                            mode='constant', constant_values=False)

    # Slice to exact shape
    sub_raw = sub_raw[:subvolume_size, :subvolume_size, :subvolume_size]
    sub_mask_1 = sub_mask_1[:subvolume_size, :subvolume_size, :subvolume_size]
    sub_mask_2 = sub_mask_2[:subvolume_size, :subvolume_size, :subvolume_size]
    sub_mask_3 = sub_mask_3[:subvolume_size, :subvolume_size, :subvolume_size]

    # We'll build an overlaid cube: shape => (H, W, 3, D)
    overlaid_cube = np.zeros((subvolume_size, subvolume_size, 3, subvolume_size), dtype=np.uint8)

    # Define colors
    side1_color = np.array([1, 0, 0], dtype=np.float32)           # Red
    side2_color = np.array([0, 0, 1], dtype=np.float32)           # Blue
    vesicles_color = np.array([0, 1, 0], dtype=np.float32)        # Green

    for z in range(subvolume_size):
        # Normalize raw slice to [0, 1]
        raw_slice = sub_raw[z].astype(np.float32)
        mn, mx = raw_slice.min(), raw_slice.max()
        if mx > mn:
            raw_slice = (raw_slice - mn) / (mx - mn)
        else:
            raw_slice = raw_slice - mn  # all zeros if mn=mx

        raw_rgb = np.stack([raw_slice]*3, axis=-1)  # shape (H, W, 3)

        # Initialize colored masks
        mask1_rgb = np.zeros_like(raw_rgb)
        mask2_rgb = np.zeros_like(raw_rgb)
        mask3_rgb = np.zeros_like(raw_rgb)

        # Overlay masks based on segmentation_type
        if segmentation_type in [1, 3, 5]:
            mask1_rgb[sub_mask_1[z]] = side1_color
        if segmentation_type in [2, 3, 5]:
            mask2_rgb[sub_mask_2[z]] = side2_color
        if segmentation_type in [4, 5]:
            mask3_rgb[sub_mask_3[z]] = vesicles_color

        # Combine masks
        combined_masks = mask1_rgb + mask2_rgb + mask3_rgb
        # Ensure that combined masks do not exceed 1
        combined_masks = np.clip(combined_masks, 0, 1)

        # Blend raw image with masks
        overlaid_image = (1 - alpha) * raw_rgb + alpha * combined_masks
        overlaid_image = np.clip(overlaid_image, 0, 1)

        # Convert to uint8
        overlaid_image = (overlaid_image * 255).astype(np.uint8)
        overlaid_cube[:, :, :, z] = overlaid_image

    return overlaid_cube

class VideoMAEDataset(Dataset):
    """
    Dataset class that uses segmented volumes (side1 & side2) and additional masks for VideoMAE pre-training.
    """
    def __init__(self, vol_data_list: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
                 synapse_df: pd.DataFrame,
                 processor,
                 segmentation_type: int,
                 subvol_size: int = 80,
                 num_frames: int = 16,
                 alpha: float = 0.3):
        """
        Args:
            vol_data_list (List[Tuple[np.ndarray, np.ndarray, np.ndarray]]): List of (raw_vol, seg_vol, add_mask_vol).
            synapse_df (pd.DataFrame): DataFrame with synapse coordinates (central, side1, side2).
            processor: Processor for VideoMAE.
            segmentation_type (int): Type of segmentation overlay (0-5).
            subvol_size (int): Size of the sub-volume to extract.
            num_frames (int): Number of frames for the model.
            alpha (float): Blending alpha for segmentation.
        """
        self.vol_data_list = vol_data_list
        self.synapse_df = synapse_df.reset_index(drop=True)
        self.processor = processor
        self.segmentation_type = segmentation_type
        self.subvol_size = subvol_size
        self.num_frames = num_frames
        self.alpha = alpha

    def __len__(self):
        return len(self.synapse_df)
class VideoMAEDataset(Dataset):
    """
    Dataset class that uses segmented volumes (side1 & side2) and additional masks for VideoMAE pre-training.
    """
    def __init__(self, vol_data_dict: dict,
                 synapse_df: pd.DataFrame,
                 processor,
                 segmentation_type: int,
                 subvol_size: int = 80,
                 num_frames: int = 16,
                 alpha: float = 0.3):
        """
        Args:
            vol_data_dict (dict): Dictionary with keys as bbox_name and values as tuples (raw_vol, seg_vol, add_mask_vol).
            synapse_df (pd.DataFrame): DataFrame with synapse coordinates (central, side1, side2).
            processor: Processor for VideoMAE.
            segmentation_type (int): Type of segmentation overlay (0-5).
            subvol_size (int): Size of the sub-volume to extract.
            num_frames (int): Number of frames for the model.
            alpha (float): Blending alpha for segmentation.
        """
        self.vol_data_dict = vol_data_dict  # Changed to dictionary
        self.synapse_df = synapse_df.reset_index(drop=True)
        self.processor = processor
        self.segmentation_type = segmentation_type
        self.subvol_size = subvol_size
        self.num_frames = num_frames
        self.alpha = alpha

    def __len__(self):
        return len(self.synapse_df)
    def __getitem__(self, idx):
        syn_info = self.synapse_df.iloc[idx]
        bbox_name = syn_info['bbox_name']  # Use bbox_name instead of bbox_index

        # Unpack the volumes using bbox_name instead of bbox_index
        raw_vol, seg_vol, add_mask_vol = self.vol_data_dict.get(bbox_name, (None, None, None))

        if raw_vol is None or seg_vol is None or add_mask_vol is None:
            # Return dummy data if volumes not found
            pixel_values = torch.zeros((self.num_frames, 3, self.subvol_size, self.subvol_size), dtype=torch.float32)
            return pixel_values, syn_info, bbox_name

        # Coordinates
        central_coord = (
            int(syn_info['central_coord_1']),
            int(syn_info['central_coord_2']),
            int(syn_info['central_coord_3'])
        )
        side1_coord = (
            int(syn_info['side_1_coord_1']),
            int(syn_info['side_1_coord_2']),
            int(syn_info['side_1_coord_3'])
        )
        side2_coord = (
            int(syn_info['side_2_coord_1']),
            int(syn_info['side_2_coord_2']),
            int(syn_info['side_2_coord_3'])
        )

        # Create the overlaid segmented cube with the additional mask based on segmentation_type
        overlaid_cube = create_segmented_cube(
            raw_vol=raw_vol,
            seg_vol=seg_vol,
            add_mask_vol=add_mask_vol,
            central_coord=central_coord,
            side1_coord=side1_coord,
            side2_coord=side2_coord,
            segmentation_type=self.segmentation_type,
            subvolume_size=self.subvol_size,
            alpha=self.alpha
        )  # shape: (80, 80, 3, 80)

        # We interpret the last dimension (depth) as frames
        frames = []
        for z in range(overlaid_cube.shape[3]):  # 80 slices
            frame_rgb = overlaid_cube[..., z]  # (80, 80, 3)
            frames.append(frame_rgb)

        # Now reduce or expand to self.num_frames
        total_slices = len(frames)  # 80
        if total_slices < self.num_frames:
            while len(frames) < self.num_frames:
                frames.append(frames[-1])
        elif total_slices > self.num_frames:
            indices = np.linspace(0, total_slices - 1, self.num_frames, dtype=int)
            frames = [frames[i] for i in indices]

        # Process using the VideoMAEImageProcessor
        inputs = self.processor(frames, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)  # (num_frames, 3, H, W)
        pixel_values = pixel_values.float()

        # Return pixel values, the corresponding DataFrame row, and the bbox name
        return pixel_values, syn_info, bbox_name  # Return the pixel values, DataFrame row, and bbox_name

def parse_args():
    """
    Parse command-line arguments for configurable paths and training parameters.
    """
    parser = argparse.ArgumentParser(description="VideoMAE Pre-training Script with Segmented Videos and Additional Masks")

    # Data directories
    parser.add_argument('--raw_base_dir', type=str, default='raw', help='Path to raw data directory')
    parser.add_argument('--seg_base_dir', type=str, default='seg', help='Path to segmentation data directory')
    parser.add_argument('--add_mask_base_dir', type=str, default='', help='Path to additional masks directory')
    parser.add_argument('--bbox_name', type=str, default='[bbox1]', help='Name of the bounding box directory')
    parser.add_argument('--excel_file', type=str, default='', help='Excel file with synapse coordinates')
    # Output and logging directories
    parser.add_argument('--csv_output_dir', type=str, default='csv_outputs', help='Directory to save CSV outputs')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save model checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory for TensorBoard logs')
    parser.add_argument('--size',type=tuple,default=(80,80),help='Size of the image')
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='Weight decay for optimizer')
    parser.add_argument('--subvol_size', type=int, default=80, help='Size of the sub-volume to extract')
    parser.add_argument('--num_frames', type=int, default=16, help='Number of frames per video clip')
    parser.add_argument('--mask_ratio', type=float, default=0.75, help='Mask ratio for VideoMAE')
    parser.add_argument('--patience', type=int, default=3, help='Patience for early stopping')
    parser.add_argument('--resume_checkpoint', type=str, default=None, help='Path to resume checkpoint')

    # GIF saving parameters
    parser.add_argument('--save_gifs_dir', type=str, default='gifs', help='Directory to save sample GIFs')
    parser.add_argument('--num_gifs', type=int, default=10, help='Number of sample GIFs to save')
    parser.add_argument('--alpha', type=float, default=0.3, help='Transparency factor for segmentation overlay')
    # New argument for segmentation type
    parser.add_argument('--segmentation_type', type=int, default=5, choices=range(0, 6),
                        help='Type of segmentation overlay:\n'
                             '0 = Raw image\n'
                             '1 = Raw + Side1\n'
                             '2 = Raw + Side2\n'
                             '3 = Raw + Side1 + Side2\n'
                             '4 = Raw + Vesicles\n'
                             '5 = Raw + Side1 + Side2 + Vesicles')

    args, _ = parser.parse_known_args()
    return args


def main(args):
    # Initialize processor
    processor = SimpleVideoProcessor(size=(80, 80))

    # List of all bboxes
    # bboxes = [f"bbox{i}" for i in range(1, 8)]  # bbox1 to bbox7
    bboxes =args.bbox_name # bbox1 to bbox7

    # Load volumes for all bboxes
    vol_data_dict = {}
    for bbox_name in args.bbox_name:
        print(bbox_name)
        raw_vol, seg_vol, add_mask_vol = load_volumes(
            bbox_name=bbox_name,
            raw_base_dir=args.raw_base_dir,
            seg_base_dir=args.seg_base_dir,
            add_mask_base_dir=args.add_mask_base_dir
        )
        # print(raw_vol)
        if raw_vol is not None and seg_vol is not None and add_mask_vol is not None:
            vol_data_dict[bbox_name] = (raw_vol, seg_vol, add_mask_vol)
        else:
            print(f"Skipping {bbox_name} due to missing volumes.")

    # Load synapse data for all bboxes
    synapse_dfs = []
    for bbox_name in bboxes:
        excel_file_path = os.path.join(args.excel_file, f"{bbox_name}.xlsx")
        if os.path.exists(excel_file_path):
            df = pd.read_excel(excel_file_path)
            df['bbox_name'] = bbox_name  # Add bbox_name column
            synapse_dfs.append(df)
        else:
            print(f"Excel file not found for {bbox_name}. Skipping.")
    syn_df = pd.concat(synapse_dfs, ignore_index=True)

    # Create dataset
    dataset = VideoMAEDataset(
        vol_data_dict=vol_data_dict,
        synapse_df=syn_df,
        processor=processor,
        segmentation_type=args.segmentation_type,
        subvol_size=args.subvol_size,
        num_frames=args.num_frames,
        alpha=args.alpha
    )

    # Process cubes and collect synapse data
    cubes = []
    syn_info_list = []  # List to collect synapse information

    for idx in range(len(dataset)):
        pixel_values, syn_info,bbox_name = dataset[idx]
        cubes.append(pixel_values)

        # Collect synapse info
        syn_info_list.append(syn_info)

    # Merge all synapse info into a single DataFrame
    merged_syn_info = pd.DataFrame(syn_info_list)

    print(f"Processed {len(cubes)} cubes successfully.")
    return cubes, merged_syn_info


if __name__ == "__main__":
    args = parse_args()
    main(args)


# args = parse_args()
# cubes = main(args)
# print(f"Processed {len(cubes)} cubes successfully.")
