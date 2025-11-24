from typing import Dict
import torch
import numpy as np
import copy
from maniflow.common.pytorch_util import dict_apply
from maniflow.common.replay_buffer import ReplayBuffer
from maniflow.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from maniflow.model.common.normalizer import LinearNormalizer
from maniflow.dataset.base_dataset import BaseDataset
from maniflow.model.vision_3d.point_process import PointCloudColorJitter
import random
from termcolor import cprint

class RoboTwinDataset(BaseDataset):
    def __init__(self,
            zarr_path, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            task_name=None,
            use_pc_color=False,
            pointcloud_color_aug_cfg=None,
            **kwargs
            ):
        super().__init__()
        self.task_name = task_name
        self.pointcloud_color_aug_cfg = pointcloud_color_aug_cfg
        cprint(f'Loading RoboTwinDataset from {zarr_path}', 'green')

        self.use_pc_color = use_pc_color
        if self.use_pc_color and self.pointcloud_color_aug_cfg is not None:
            self.aug_color = self.pointcloud_color_aug_cfg['aug_color']
            if self.aug_color:
                aug_color_params = self.pointcloud_color_aug_cfg['params']
                self.aug_prob = self.pointcloud_color_aug_cfg['prob']
                self.pc_jitter = PointCloudColorJitter(
                    brightness=aug_color_params[0],
                    contrast=aug_color_params[1], 
                    saturation=aug_color_params[2],
                    hue=aug_color_params[3]
                )
                cprint(f'Apply point cloud color jitter with params: {aug_color_params} and prob: {self.aug_prob}', 'red')
        else:
            self.aug_color = False

        buffer_keys = [
            'point_cloud', # default use point_cloud
            'state', 
            'action',]

        self.replay_buffer = ReplayBuffer.copy_from_path(
                zarr_path, keys=buffer_keys)

        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask

        if max_train_episodes is None:
            # Use all training episodes
            max_train_episodes = self.replay_buffer.n_episodes - np.sum(val_mask)
        cprint(f'Maximum training episodes: {max_train_episodes}', 'yellow')
        cprint(f'Validation ratio: {val_ratio}', 'yellow')

        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

        self.zarr_path = zarr_path
        self.train_episodes_num = np.sum(train_mask)
        self.val_episodes_num = np.sum(val_mask)

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    
    def get_normalizer(self, mode='limits', **kwargs):
       
        data = {
            'point_cloud': self.replay_buffer['point_cloud'],
            'agent_pos': self.replay_buffer['state'][...,:],
            'action': self.replay_buffer['action']
            }
        
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        
        return normalizer


    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        agent_pos = sample['state'][:,].astype(np.float32)
        point_cloud = sample['point_cloud'][:,].astype(np.float32) # (T, 512, 3)

        data = {
            'obs': {
                'point_cloud': point_cloud,
                'agent_pos': agent_pos,
                },
            'action': sample['action'].astype(np.float32)}

        
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)

        # point cloud color jitter augmentation
        if self.aug_color:
            if random.random() > self.aug_prob:
                return torch_data
            else:
                T, N, C = torch_data['obs']['point_cloud'].shape
                pc_reshaped = torch_data['obs']['point_cloud'].reshape(-1, C)
                pc_reshaped = self.pc_jitter(pc_reshaped)
                torch_data['obs']['point_cloud'] = pc_reshaped.reshape(T, N, C) 

        return torch_data

if __name__ == '__main__':
    # Test dataset
    zarr_path = '/path/to/robotwin_dataset.zarr'  # Replace with actual path
    dataset = RoboTwinDataset(zarr_path=zarr_path, 
                            horizon=1, 
                            pad_before=0, 
                            pad_after=0,)
    print(f"Dataset size: {len(dataset)}")
    print(f"Train episodes: {dataset.train_episodes_num}")
    print(f"Validation episodes: {dataset.val_episodes_num}")
    print(f"Task name: {dataset.task_name}")

    # Test sampling
    sample = dataset[0]
    print("\nSample structure:")
    for key, value in sample.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for subkey, subvalue in value.items():
                print(f"  {subkey}: shape={subvalue.shape if hasattr(subvalue, 'shape') else 'scalar'}")
        else:
            print(f"{key}: shape={value.shape if hasattr(value, 'shape') else 'scalar'}")

    # Test normalizer
    normalizer = dataset.get_normalizer()
    print("\nNormalizer stats:")
    print(normalizer.get_input_stats())
    print(normalizer.get_output_stats())
    
    # Test validation set
    val_dataset = dataset.get_validation_dataset()
    print(f"\nValidation dataset size: {len(val_dataset)}")