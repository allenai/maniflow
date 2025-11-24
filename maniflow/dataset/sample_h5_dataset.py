import os
from pathlib import Path
import h5py
import numpy as np
from decord import VideoReader, cpu
import decord
import torch
from collections import defaultdict
import json
import logging
from typing import Dict
import copy

logger = logging.getLogger(__name__)



def pad_data(data, start_step: int, end_step: int, data_start: int, data_end: int):
    """
    Args:
        data: The actual loaded data (could be subset of trajectory)
        start_step: Desired start index in trajectory coordinates
        end_step: Desired end index in trajectory coordinates (exclusive)
        data_start: Index of the trajectory where data[0] corresponds to
        data_end: Index of the trajectory where data ends (exclusive, so data[-1] is at data_end-1)
    """
    window_size = end_step - start_step
    is_pad = torch.zeros(window_size, dtype=torch.bool)

    # Mark which positions in the window need padding
    for i, traj_idx in enumerate(range(start_step, end_step)):
        if traj_idx < data_start or traj_idx >= data_end:
            is_pad[i] = True

    # If no padding needed, just return the data
    if not any(is_pad):
        return data, is_pad

    # Calculate how much padding we need
    front_pad_length = max(0, data_start - start_step)
    back_pad_length = max(0, end_step - data_end)

    # Create padding arrays
    pad_shape_front = (front_pad_length,) + data.shape[1:]
    pad_shape_back = (back_pad_length,) + data.shape[1:]

    front_padding = np.zeros(pad_shape_front, dtype=np.float32)
    back_padding = np.zeros(pad_shape_back, dtype=np.float32)

    # Concatenate: [front_padding, data, back_padding]
    padded_data = np.concatenate([front_padding, data, back_padding], axis=0).astype(np.float32)

    return padded_data, is_pad


from maniflow.dataset.base_dataset import BaseDataset
from maniflow.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from maniflow.common.pytorch_util import dict_apply


class MjThorToSpocDataset(BaseDataset):
    """Dataset for MjThor -> Spoc format data.
    
    This dataset supports two different modes for train/val splitting:
    
    1. SPLIT SUBDIRECTORIES (Recommended):
       - Organize your data as: data_path/train/ and data_path/val/
       - Instantiate separate datasets: MjThorToSpocDataset(..., split="train") 
         and MjThorToSpocDataset(..., split="val")
       - Each dataset will load ALL data from its respective subdirectory
       - No random splitting occurs
    
    2. RANDOM SPLITTING (Fallback):
       - If split subdirectories don't exist, all data from data_path is loaded
       - Data is randomly split into train/val based on val_ratio
       - Use get_validation_dataset() to get the val split
       - Only use this mode if you can't organize data into subdirectories
    
    Args:
        data_path: Path to the data directory (base path, without /train or /val)
        split: Which split to load - "train", "val", or "test" (default: "train")
        val_ratio: Validation split ratio (only used in fallback mode, default: 0.02)
        seed: Random seed for splitting (only used in fallback mode, default: 42)
    """
    def __init__(
        self, 
        data_path: str,
        camera_names: list[str] = [],
        action_move_group_names: list[str] = [],
        action_spec: dict = {},
        input_window_size: int = 4,
        action_chunk_size: int = 8,
        val_ratio: float = 0.02,
        seed: int = 42,
        split: str = "train",  # "train", "val", or "test"
    ):
        super().__init__()
        
        # Store paths and params
        self.base_path = Path(data_path)
        self.split = split
        self.val_ratio = val_ratio
        self.seed = seed
        
        # Input Params
        self.camera_names = camera_names
        self.input_window_size = input_window_size

        # Output Params
        self.action_move_group_names = action_move_group_names
        self.action_spec = action_spec
        self.action_dim = sum(self.action_spec[mg] for mg in self.action_move_group_names)
        self.action_chunk_size = action_chunk_size
        
        # Determine whether to use split subdirectories or random splitting
        # Check if train/ subdirectory exists (val/ is optional)
        train_subdir_exists = (self.base_path / "train").exists()
        val_subdir_exists = (self.base_path / "val").exists()
        split_path = self.base_path / split
        
        # If train/ subdirectory exists, use split subdirectory mode
        if train_subdir_exists:
            # Check if val has data
            val_has_data = False
            if val_subdir_exists:
                val_path = self.base_path / "val"
                val_test_files = []
                if val_path.exists():
                    for house_dir in os.listdir(val_path):
                        if os.path.isdir(val_path / house_dir):
                            for file in os.listdir(val_path / house_dir):
                                if file.endswith("h5"):
                                    val_has_data = True
                                    break
                            if val_has_data:
                                break
            
            # Decide on mode based on whether val has data
            if val_has_data:
                # Val has its own data - use split subdirectory mode
                self.use_split_subdirs = True
                
                if split == "train":
                    self.data_path = self.base_path / "train"
                    self._files = self._get_traj_files()
                    
                    if len(self._files) == 0:
                        raise ValueError(f"Train subdirectory exists but contains no data files: {self.data_path}")
                    
                    logger.info(f"Using train split subdirectory: {self.data_path} (found {len(self._files)} files)")
                
                elif split == "val":
                    self.data_path = self.base_path / "val"
                    self._files = self._get_traj_files()
                    logger.info(f"Using val split subdirectory: {self.data_path} (found {len(self._files)} files)")
                
                else:
                    # test or other splits
                    if split_path.exists():
                        self.data_path = split_path
                        self._files = self._get_traj_files()
                        logger.info(f"Using {split} split subdirectory: {self.data_path} (found {len(self._files)} files)")
                    else:
                        logger.warning(f"{split} subdirectory not found. Dataset will be empty.")
                        self.data_path = split_path
                        self._files = []
            else:
                # Val doesn't exist or is empty - split train data for both train and val
                self.use_split_subdirs = False
                self.data_path = self.base_path / "train"
                self._files = self._get_traj_files()
                
                if len(self._files) == 0:
                    raise ValueError(f"Train subdirectory exists but contains no data files: {self.data_path}")
                
                logger.info(f"Val split not found or empty. Using train data with random splitting: {self.data_path} (found {len(self._files)} files)")
        else:
            # No train/ subdirectory exists, use base path with random splitting
            self.use_split_subdirs = False
            self.data_path = self.base_path
            self._files = self._get_traj_files()
            
            if len(self._files) == 0:
                raise ValueError(f"No data files found in base path: {self.base_path}")
            
            logger.info(f"No split subdirectories found, using base path with random splitting: {self.base_path} (found {len(self._files)} files)")

        # Internal bookkeeping
        self.traj_idx_to_file_and_traj = {}
        self.traj_idx_to_length = {}
        self.traj_indices = []
        self.traj_lengths = []
        self.traj_cumsum_lengths = None
        self._build_trajectory_bookkeeping()
        
        # Create train/val split based on mode
        if self.use_split_subdirs:
            # Use all data from this split subdirectory - no further splitting needed
            self.train_indices = self.traj_indices
            self.val_indices = []
            # Set is_train based on which split this is
            self.is_train = (self.split == "train")
        else:
            # Fallback: randomly split the data from base path
            self._create_train_val_split()

        # Decord
        decord.bridge.set_bridge("torch")
        self.decord_ctx = cpu(0)
        
        # Aliases for workspace compatibility
        self.zarr_path = str(self.data_path)  # Workspace expects this
        if self.use_split_subdirs:
            # When using split subdirs, report all trajectories as belonging to this split
            if self.split == "train":
                self.train_episodes_num = len(self.traj_indices)
                self.val_episodes_num = 0
            else:
                self.train_episodes_num = 0
                self.val_episodes_num = len(self.traj_indices)
        else:
            # When using random splitting, use the split indices
            self.train_episodes_num = len(self.train_indices)
            self.val_episodes_num = len(self.val_indices)


    
    def __len__(self):
        # If using split subdirectories, use all data from this split
        if self.use_split_subdirs:
            if len(self.traj_indices) == 0:
                return 0
            return sum(self.traj_idx_to_length[idx] for idx in self.traj_indices)
        
        # Otherwise, use the appropriate indices based on train/val split
        active_indices = self.train_indices if self.is_train else self.val_indices
        if len(active_indices) == 0:
            return 0
        
        # Sum up lengths of active trajectories
        total_length = sum(self.traj_idx_to_length[idx] for idx in active_indices)
        return total_length

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        global_traj_idx, step = self._flat_idx_to_traj_idx(idx)
        file_idx, traj_idx = self._get_file_and_traj_idx(global_traj_idx)

        # Collect observation data
        obs = {}
        obs.update(self._get_camera_frames(file_idx, traj_idx, step))
        obs.update(self._get_last_actions(file_idx, traj_idx, step))
        obs.update(self._get_time_ids())
        obs.update(self._get_traj_index(global_traj_idx))
        obs.update(self._get_goal(file_idx, traj_idx))
        
        # Get action data
        action_data = self._get_actions(file_idx, traj_idx, step)
        
        # Format in ManiFlow style: {'obs': {...}, 'action': ...}
        data = {
            'obs': obs,
            'action': action_data['actions'],
            'action_is_pad': action_data['actions_is_pad']
        }
        
        return data

    def _get_traj_files(self):
        files = []
        for house_dir in os.listdir(self.data_path):
            if os.path.isdir(self.data_path / house_dir):
                for file in os.listdir(self.data_path / house_dir):
                    if file.endswith("h5"):
                        files.append(self.data_path / house_dir / file)
        return sorted(files)  # Sort for consistent ordering
    
    def _build_trajectory_bookkeeping(self):
        global_traj_idx = 0
        
        for file_idx, file_path in enumerate(self._files):
            try:
                with h5py.File(file_path, "r") as file:
                    traj_keys = [key for key in file.keys() if key.startswith("traj_")]
                    traj_keys = sorted(traj_keys, key=lambda x: int(x.split("_")[1]))
                    
                    for traj_key in traj_keys:
                        traj_idx = int(traj_key.split("_")[1])
                        traj_length = file[traj_key]['success'].shape[0] - 1 # get length of trajectory (minus 1 because we don't include the first action)
                        
                        if traj_length > 0:
                            self.traj_idx_to_file_and_traj[global_traj_idx] = (file_idx, traj_idx)
                            self.traj_idx_to_length[global_traj_idx] = traj_length
                            self.traj_indices.append(global_traj_idx)
                            self.traj_lengths.append(traj_length)
                            
                            global_traj_idx += 1
            except OSError as e:
                logger.warning(f"Skipping corrupted file {file_path}: {e}")
                continue
            except Exception as e:
                logger.warning(f"Skipping file {file_path} due to error: {e}")
                continue
        
        if len(self.traj_lengths) > 0:
            self.traj_cumsum_lengths = np.cumsum(self.traj_lengths)
        else:
            self.traj_cumsum_lengths = np.array([])
    
    def _create_train_val_split(self):
        """Create train/validation split based on trajectories."""
        num_trajectories = len(self.traj_indices)
        
        if num_trajectories == 0:
            self.train_indices = []
            self.val_indices = []
            self.is_train = (self.split == "train")
            return
        
        # Set random seed for reproducibility
        rng = np.random.RandomState(self.seed)
        
        # Shuffle trajectory indices
        shuffled_indices = rng.permutation(num_trajectories)
        
        # Split into train and val
        num_val = max(1, int(num_trajectories * self.val_ratio))
        val_traj_positions = shuffled_indices[:num_val]
        train_traj_positions = shuffled_indices[num_val:]
        
        self.train_indices = [self.traj_indices[i] for i in train_traj_positions]
        self.val_indices = [self.traj_indices[i] for i in val_traj_positions]
        # Respect the split parameter - if split="val", this is the validation set
        self.is_train = (self.split == "train")
    
    def _flat_idx_to_traj_idx(self, flat_idx):
        # If using split subdirectories, use all trajectories from this split
        if self.use_split_subdirs:
            active_indices = self.traj_indices
        else:
            # Otherwise, use appropriate indices based on train/val split
            active_indices = self.train_indices if self.is_train else self.val_indices
        
        if len(active_indices) == 0:
            raise IndexError(f"Dataset is empty, cannot access index {flat_idx}")
        
        # Build cumulative lengths for active trajectories only
        active_lengths = [self.traj_idx_to_length[idx] for idx in active_indices]
        active_cumsum = np.cumsum(active_lengths)
        
        # Find which trajectory this flat index belongs to
        traj_position = np.searchsorted(active_cumsum, flat_idx, side="right")
        
        if traj_position > 0:
            step = flat_idx - active_cumsum[traj_position - 1]
        else:
            step = flat_idx
        
        global_traj_idx = active_indices[traj_position]
        return global_traj_idx, step
    
    def _get_file_and_traj_idx(self, global_traj_idx) -> tuple[int, int]:
        if global_traj_idx not in self.traj_idx_to_file_and_traj:
            raise ValueError(f"Global trajectory index {global_traj_idx} not found")
        
        file_idx, traj_idx = self.traj_idx_to_file_and_traj[global_traj_idx]
        return file_idx, traj_idx
    def _decode_dict_data(self, traj_idx, keys, data):

        # Load and decode data if not cached
        all_dict_data = {}
        for key in keys:
            key_data = data[key]
            trajectories = []
            for i in range(key_data.shape[0]):
                byte_array = key_data[i]
                json_string = byte_array.tobytes().decode('utf-8').rstrip('\x00')
                trajectory_dict = json.loads(json_string)
                trajectories.append(trajectory_dict)
            all_dict_data[key] = trajectories
                
        return all_dict_data
    
    def _get_time_ids(self):
        return {"time_ids": torch.arange(self.input_window_size)}
    
    def _get_traj_index(self, global_traj_idx: int) -> dict[str, torch.Tensor]:
        return {"traj_index": torch.full((self.input_window_size,), global_traj_idx, dtype=torch.long)}
    
    def _get_camera_frames(self, file_idx: int, traj_idx: int, step: int) -> dict[str, torch.Tensor]:
        sample = {}
        file_path = self._files[file_idx]
        traj_key = f"traj_{traj_idx}"

        try:
            with h5py.File(file_path, "r") as file:
                window_start = step - self.input_window_size + 1
                window_end = step + 1

                for camera_name in self.camera_names:
                    try:
                        frames = []
                        vr = None  # Will load video once
                        
                        for i in range(window_start, window_end):
                            if i < 0:
                                # Lazy load video to get shape
                                if vr is None:
                                    obs_data = file[traj_key]["obs"]["sensor_data"][camera_name]
                                    video_filename = obs_data[:].tobytes().decode('utf-8').rstrip('\x00')
                                    video_path = str(self.data_path / file_path.parent / video_filename)
                                    vr = VideoReader(video_path, ctx=self.decord_ctx)
                                
                                # Get shape from first frame
                                H, W, C = vr[0].shape
                                # Append zero-padded frame (as float for consistency with normalized frames)
                                frames.append(torch.zeros(H, W, C, dtype=torch.float32))
                            else:
                                # Load video if not already loaded
                                if vr is None:
                                    obs_data = file[traj_key]["obs"]["sensor_data"][camera_name]
                                    video_filename = obs_data[:].tobytes().decode('utf-8').rstrip('\x00')
                                    video_path = str(self.data_path / file_path.parent / video_filename)
                                    vr = VideoReader(video_path, ctx=self.decord_ctx)
                                
                                frame = vr[i]  # Get single frame
                                frames.append(frame)
                        
                        # Stack frames: (input_window_size, H, W, C)
                        # Normalize to [0, 1] range
                        sample[camera_name] = torch.stack(frames).float() / 255.0
                    except KeyError as e:
                        logger.warning(f"Camera '{camera_name}' not found in {file_path} trajectory {traj_idx}: {e}. Skipping this camera.")
                        continue
        except OSError as e:
            logger.error(f"Failed to open corrupted file {file_path}: {e}")
            raise RuntimeError(f"Cannot read from corrupted file {file_path}") from e
        
        return sample

    def _get_actions(self, file_idx: int, traj_idx: int, step: int) -> dict[str, torch.Tensor]:
        """Return 1 action chunk per window size timestep.
        
        Returns actions of shape (input_window_size, action_chunk_size, action_dim)
        where each timestep in the window gets one action chunk.
        """
        file_path = self._files[file_idx]

        try:
            with h5py.File(file_path, "r") as file:
                try:
                    action_data = file["traj_" + str(traj_idx)]
                except KeyError as e:
                    logger.error(f"Trajectory 'traj_{traj_idx}' not found in {file_path}: {e}")
                    raise RuntimeError(f"Trajectory 'traj_{traj_idx}' not found in {file_path}") from e
                decoded_data = self._decode_dict_data(traj_idx, ["actions"], action_data)["actions"]
                
                window_start = step - self.input_window_size + 1
                window_end = step + 1
                
                # Collect one action chunk per timestep in the window
                all_chunks = []
                all_is_pad = []
                
                for window_timestep in range(window_start, window_end):
                    # For each timestep in the window, get one action chunk
                    # Actions start from window_timestep + 1 (actions are 1-indexed)
                    chunk_start = window_timestep + 1
                    chunk_end = window_timestep + 1 + self.action_chunk_size
                    
                    traj_chunk_start = max(0, chunk_start)
                    traj_chunk_end = min(len(decoded_data), chunk_end)
                    
                    chunk_actions = []
                    for i in range(traj_chunk_start, traj_chunk_end):
                        action_vec = []
                        for move_group in self.action_move_group_names:
                            try:
                                action_vec.append(decoded_data[i][move_group])
                            except KeyError:
                                action_vec.append(np.zeros(self.action_spec[move_group]))
                        action_vec = np.concatenate(action_vec)
                        chunk_actions.append(action_vec)
                    
                    # Pad chunk if needed
                    if len(chunk_actions) == 0:
                        dummy_action = np.concatenate([np.zeros(self.action_spec[mg]) for mg in self.action_move_group_names])
                        chunk_array = np.zeros((0, len(dummy_action)), dtype=np.float32)
                    else:
                        chunk_array = np.array(chunk_actions, dtype=np.float32)
                    
                    padded_chunk, chunk_is_pad = pad_data(chunk_array, chunk_start, chunk_end, traj_chunk_start, traj_chunk_end)
                    all_chunks.append(padded_chunk)
                    all_is_pad.append(chunk_is_pad)
                
                # Stack all chunks: (input_window_size, action_chunk_size, action_dim)
                actions_tensor = torch.stack([torch.from_numpy(chunk) for chunk in all_chunks])
                # pad_data returns is_pad as torch.Tensor, so no need to convert from numpy
                is_pad_tensor = torch.stack(all_is_pad)
                
                return {"actions": actions_tensor, "actions_is_pad": is_pad_tensor}
        except OSError as e:
            logger.error(f"Failed to open corrupted file {file_path}: {e}")
            raise RuntimeError(f"Cannot read from corrupted file {file_path}") from e
    
    def _get_last_actions(self, file_idx: int, traj_idx: int, step: int) -> dict[str, torch.Tensor]:
        """Return window of previous continuous actions"""
        last_actions = []
        traj_key = f"traj_{traj_idx}"
        file_path = self._files[file_idx]

        try:
            with h5py.File(file_path, "r") as file:
                try:
                    action_data = file[traj_key]
                except KeyError as e:
                    logger.error(f"Trajectory '{traj_key}' not found in {file_path}: {e}")
                    raise RuntimeError(f"Trajectory '{traj_key}' not found in {file_path}") from e
                all_actions = self._decode_dict_data(traj_idx, ["actions"], action_data)["actions"]
                
                window_start = step - self.input_window_size + 1
                window_end = step + 1
                
                for i in range(window_start, window_end):
                    if i <= 0:
                        last_actions.append(torch.zeros(self.action_dim))
                    else:
                        action_dict = all_actions[i]
                        action_tensors = []
                        for action_move_group_name in self.action_move_group_names:
                            if action_move_group_name not in action_dict:
                                logger.warning(f"Action move group {action_move_group_name} not found. Padding with zeros.")
                                action_tensors.append(torch.zeros(self.action_spec[action_move_group_name]))
                            else:
                                action_tensors.append(torch.tensor(action_dict[action_move_group_name]))
                        last_actions.append(torch.cat(action_tensors))
        except OSError as e:
            logger.error(f"Failed to open corrupted file {file_path}: {e}")
            raise RuntimeError(f"Cannot read from corrupted file {file_path}") from e
        
        # Stack into (input_window_size, action_dim) tensor
        return {"last_actions": torch.stack(last_actions)}

    def _get_goal(self, file_idx: int, traj_idx: int) -> dict[str, str]:
        file_path = self._files[file_idx]
        sample = {}
        try:
            with h5py.File(file_path, "r") as file:
                traj_key = f"traj_{traj_idx}"
                try:
                    obs_scene_data = file[traj_key]['obs_scene'][()]
                    # Handle both bytes and numpy array cases
                    if isinstance(obs_scene_data, bytes):
                        json_string = obs_scene_data.decode('utf-8').rstrip('\x00')
                    else:
                        json_string = obs_scene_data.tobytes().decode('utf-8').rstrip('\x00')
                    goal = json.loads(json_string)['task_description']
                    sample["goal"] = goal
                except KeyError as e:
                    logger.error(f"Key 'obs_scene' not found in trajectory '{traj_key}' in {file_path}: {e}")
                    raise RuntimeError(f"Key 'obs_scene' not found in trajectory '{traj_key}' in {file_path}") from e
        except OSError as e:
            logger.error(f"Failed to open corrupted file {file_path}: {e}")
            raise RuntimeError(f"Cannot read from corrupted file {file_path}") from e
        return sample
    
    def get_normalizer(self, mode='limits', **kwargs):
        """Create normalizer for the dataset following ManiFlow pattern."""
        normalizer = LinearNormalizer()
        
        # For images/cameras, use identity normalization (already normalized to [0, 1])
        for camera_name in self.camera_names:
            normalizer[camera_name] = SingleFieldLinearNormalizer.create_identity()
        
        # For actions, we would need to collect all actions from the dataset
        # For now, use identity normalization
        normalizer['action'] = SingleFieldLinearNormalizer.create_identity()
        normalizer['last_actions'] = SingleFieldLinearNormalizer.create_identity()
        
        return normalizer
    
    def get_validation_dataset(self):
        """Return a validation split of the dataset.
        
        Note: If using split subdirectories (train/, val/), you should instantiate
        a separate dataset with split="val" instead of calling this method.
        This method only works when using random splitting (fallback mode).
        """
        if self.use_split_subdirs:
            logger.warning(
                "get_validation_dataset() called when using split subdirectories. "
                "You should instantiate a separate dataset with split='val' instead. "
                "Returning a copy with is_train=False, but this may not work as expected."
            )
        
        val_dataset = copy.deepcopy(self)
        val_dataset.is_train = False
        return val_dataset
    
    def max_seq_len(self, tokenize_actions, num_samples=500):
        import random
        from collections import Counter

        total_samples = len(self)
        sample_size = min(num_samples, total_samples)
        sampled_indices = random.sample(range(total_samples), sample_size)

        token_lengths = []
        logger.info(f"Tokenizing {sample_size} samples...")

        for idx in sampled_indices:
            sample = self[idx]
            action = sample["action"]

            tokens = tokenize_actions(action)
            for i in range(self.input_window_size):
                token_lengths.append(len(tokens[i]))

        max_length = max(token_lengths)
        min_length = min(token_lengths)
        avg_length = sum(token_lengths) / len(token_lengths)

        length_counter = Counter(token_lengths)
        sorted_lengths = sorted(length_counter.keys())

        logger.info(f"\n{'='*60}")
        logger.info(f"Token Length Statistics (n={sample_size})")
        logger.info(f"{'='*60}")
        logger.info(f"Min length: {min_length}")
        logger.info(f"Max length: {max_length}")
        logger.info(f"Avg length: {avg_length:.2f}")
        logger.info(f"\nHistogram of Token Lengths:")
        logger.info(f"{'Length':<10} {'Count':<10} {'Distribution'}")
        logger.info(f"{'-'*60}")

        for length in sorted_lengths:
            count = length_counter[length]
            bar = '█' * int((count / sample_size) * 50)
            logger.info(f"{length:<10} {count:<10} {bar}")

        logger.info(f"{'='*60}\n")

        return max_length       

if __name__ == "__main__":
    # Example usage
    dataset = MjThorToSpocDataset(
        data_path="/path/to/data",
        camera_names=["wrist_camera_r", "head_camera"],
        action_move_group_names=["base", "right_arm", "right_gripper"],
        action_spec={
            "base": 3,
            "head": 2,
            "right_arm": 7,
            "left_arm": 7,
            "right_gripper": 2,
            "left_gripper": 2,
        },
        input_window_size=4,
        action_chunk_size=8,
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Action dimension: {dataset.action_dim}")
    
    if len(dataset) > 0:
        sample = dataset[0]
        print("\nSample structure:")
        for key, value in sample.items():
            if isinstance(value, dict):
                print(f"{key}:")
                for subkey, subvalue in value.items():
                    print(f"  {subkey}: shape={subvalue.shape if hasattr(subvalue, 'shape') else type(subvalue)}")
            else:
                print(f"{key}: shape={value.shape if hasattr(value, 'shape') else type(value)}")