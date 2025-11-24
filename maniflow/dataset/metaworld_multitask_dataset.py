from typing import Dict, List
import numpy as np
import copy
from pathlib import Path
from termcolor import cprint
from maniflow.common.replay_buffer import ReplayBuffer
from maniflow.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from maniflow.model.common.normalizer import LinearNormalizer
from maniflow.dataset.base_dataset import BaseDataset

class MetaworldMultitaskDataset(BaseDataset):
    def __init__(self,
            data_path: str,
            task_names: List[str],
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            ):
        super().__init__()
        
        self.task_names = task_names
        self.replay_buffers = {}
        self.samplers = {}
        self.train_masks = {}
        self.train_episodes_num = 0
        self.val_episodes_num = 0
        
        # Load data for each task
        for task_name in task_names:
            zarr_path = Path(data_path) / f"metaworld_{task_name}_expert.zarr"
            rb = ReplayBuffer.copy_from_path(
                zarr_path, keys=['state', 'action', 'point_cloud'])
            
            val_mask = get_val_mask(
                n_episodes=rb.n_episodes,
                val_ratio=val_ratio,
                seed=seed)
            train_mask = ~val_mask
            train_mask = downsample_mask(
                mask=train_mask,
                max_n=max_train_episodes,
                seed=seed)
            
            sampler = SequenceSampler(
                replay_buffer=rb,
                sequence_length=horizon,
                pad_before=pad_before,
                pad_after=pad_after,
                episode_mask=train_mask)

            self.replay_buffers[task_name] = rb
            self.samplers[task_name] = sampler
            self.train_masks[task_name] = train_mask
            self.train_episodes_num += np.sum(train_mask)
            self.val_episodes_num += np.sum(val_mask)

            cprint(f"Task {task_name}: {np.sum(train_mask)} training episodes and {len(sampler)} rollout steps", 'green')

        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.data_path = data_path
        
    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.samplers = {}
        for task_name in self.task_names:
            val_set.samplers[task_name] = SequenceSampler(
                replay_buffer=self.replay_buffers[task_name],
                sequence_length=self.horizon,
                pad_before=self.pad_before,
                pad_after=self.pad_after,
                episode_mask=~self.train_masks[task_name]
            )
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        # Combine data from all tasks
        all_actions = []
        all_agent_pos = []
        all_point_clouds = []
        
        for task_name in self.task_names:
            rb = self.replay_buffers[task_name]
            all_actions.append(rb['action'])
            all_agent_pos.append(rb['state'][...,:])
            all_point_clouds.append(rb['point_cloud'])

        data = {
            'action': np.concatenate(all_actions),
            'agent_pos': np.concatenate(all_agent_pos),
            'point_cloud': np.concatenate(all_point_clouds),
        }
        
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def __len__(self) -> int:
        total_len = 0
        for sampler in self.samplers.values():
            total_len += len(sampler)
        return total_len

    def _sample_to_data(self, sample, task_name):
        agent_pos = sample['state'][:,].astype(np.float32)
        point_cloud = sample['point_cloud'][:,].astype(np.float32)

        # Add task ID as one-hot encoding
        task_idx = self.task_names.index(task_name)
        task_one_hot = np.zeros(len(self.task_names), dtype=np.float32)
        task_one_hot[task_idx] = 1.0

        data = {
            'obs': {
                'point_cloud': point_cloud,
                'agent_pos': agent_pos,
                'task_id': task_one_hot,
                'task_name': task_name
            },
            'action': sample['action'].astype(np.float32)
        }
        return data

    def __getitem__(self, idx):
        # Determine which task this index belongs to
        current_idx = idx
        for task_name, sampler in self.samplers.items():
            if current_idx < len(sampler):
                sample = sampler.sample_sequence(current_idx)
                return self._sample_to_data(sample, task_name)
            current_idx -= len(sampler)
        # If we reach here, idx is out of range
        raise IndexError("Index out of range") 
    

if __name__ == "__main__":
    # Test the dataset
    data_path = "/path/to/metaworld/data"  # Replace with actual path
    task_names = ['assembly', 'basketball']
    dataset = MetaworldMultitaskDataset(
        data_path=data_path,
        task_names=task_names,
        horizon=10,
        pad_before=2,
        pad_after=2,
        seed=42,
        val_ratio=0.2
    )
    
    # Test basic properties
    print(f"Dataset size: {len(dataset)}")
    
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
    
