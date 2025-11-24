#!/usr/bin/env python3
"""
Quick test script to verify MjThorToSpoc dataset integration.
Run this before starting full training to catch issues early.

Usage:
    python ManiFlow/maniflow/config/test_mjthor_dataset.py
"""

import sys
import os
from pathlib import Path

# Add ManiFlow to path
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

import hydra
from omegaconf import OmegaConf
import torch
from maniflow.dataset.sample_h5_dataset import MjThorToSpocDataset

def test_direct_instantiation():
    """Test 1: Direct instantiation without Hydra"""
    print("=" * 70)
    print("TEST 1: Direct Dataset Instantiation")
    print("=" * 70)
    
    try:
        dataset = MjThorToSpocDataset(
            data_path="/weka/prior/datasets/robomolmo/testing_door_14NOV/DoorOpeningDataGenConfig",
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
            split="train",
        )
        
        print(f"✓ Dataset created successfully")
        print(f"  - Total samples: {len(dataset)}")
        print(f"  - Number of trajectories: {len(dataset.traj_indices)}")
        print(f"  - Action dimension: {dataset.action_dim}")
        
        if len(dataset) > 0:
            print(f"\n✓ Attempting to load first sample...")
            sample = dataset[0]
            
            print(f"\n✓ Sample loaded successfully!")
            print(f"\nSample structure:")
            print(f"  obs keys: {list(sample['obs'].keys())}")
            for key, value in sample['obs'].items():
                if torch.is_tensor(value):
                    print(f"    - {key}: {value.shape} ({value.dtype})")
                else:
                    print(f"    - {key}: {type(value)} = {value}")
            
            print(f"\n  action: {sample['action'].shape} ({sample['action'].dtype})")
            print(f"  action_is_pad: {sample['action_is_pad'].shape} ({sample['action_is_pad'].dtype})")
            print(f"    - Padded timesteps: {sample['action_is_pad'].sum().item()}/{sample['action_is_pad'].numel()}")
            
            # Check value ranges
            print(f"\n✓ Checking value ranges:")
            for cam in dataset.camera_names:
                if cam in sample['obs']:
                    cam_data = sample['obs'][cam]
                    print(f"    - {cam}: min={cam_data.min():.3f}, max={cam_data.max():.3f}")
            
            action_data = sample['action']
            print(f"    - action: min={action_data.min():.3f}, max={action_data.max():.3f}")
            
            print(f"\n✓ Test 1 PASSED!")
            return True
        else:
            print(f"\n✗ Dataset is empty!")
            print(f"  Check that:")
            print(f"    1. data_path exists: {dataset.data_path}")
            print(f"    2. H5 files are in subdirectories (e.g., data_path/house_name/*.h5)")
            print(f"    3. H5 files found: {len(dataset._files)}")
            if len(dataset._files) > 0:
                print(f"    4. Files found but no valid trajectories detected")
                print(f"       - Check that H5 files contain 'traj_*' groups with 'success' field")
            return False
            
    except Exception as e:
        print(f"\n✗ Test 1 FAILED!")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_hydra_config():
    """Test 2: Load via Hydra config"""
    print("\n" + "=" * 70)
    print("TEST 2: Hydra Config Loading")
    print("=" * 70)
    
    try:
        # Initialize Hydra - load the main config, not the custom_task directly
        hydra.initialize(config_path=".", version_base="1.2")
        cfg = hydra.compose(config_name="maniflow_image_mjthor_spoc")
        
        print(f"✓ Config loaded successfully")
        
        # Dataset is nested under custom_task in the config structure
        if hasattr(cfg, 'custom_task') and hasattr(cfg.custom_task, 'dataset'):
            dataset_cfg = cfg.custom_task.dataset
        elif hasattr(cfg, 'dataset'):
            dataset_cfg = cfg.dataset
        else:
            raise ValueError("Could not find dataset config in cfg or cfg.custom_task")
        
        print(f"\nDataset config:")
        print(OmegaConf.to_yaml(dataset_cfg))
        
        # Try to instantiate
        print(f"\n✓ Attempting to instantiate via Hydra...")
        dataset = hydra.utils.instantiate(dataset_cfg)
        
        print(f"✓ Dataset instantiated via Hydra!")
        print(f"  - Total samples: {len(dataset)}")
        
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        
        print(f"\n✓ Test 2 PASSED!")
        return True
        
    except Exception as e:
        print(f"\n✗ Test 2 FAILED!")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        try:
            hydra.core.global_hydra.GlobalHydra.instance().clear()
        except:
            pass
        return False


def test_dataloader():
    """Test 3: Create DataLoader and iterate"""
    print("\n" + "=" * 70)
    print("TEST 3: DataLoader Integration")
    print("=" * 70)
    
    try:
        dataset = MjThorToSpocDataset(
            data_path="/weka/prior/datasets/robomolmo/testing_door_14NOV/DoorOpeningDataGenConfig",
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
            split="train",
        )
        
        if len(dataset) == 0:
            print(f"✗ Dataset is empty, skipping dataloader test")
            return False
        
        from torch.utils.data import DataLoader
        
        dataloader = DataLoader(
            dataset,
            batch_size=4,
            num_workers=0,  # Use 0 for testing to avoid multiprocessing issues
            shuffle=False,
        )
        
        print(f"✓ DataLoader created successfully")
        print(f"  - Batch size: 4")
        print(f"  - Total batches: {len(dataloader)}")
        
        # Load one batch
        print(f"\n✓ Loading first batch...")
        batch = next(iter(dataloader))
        
        print(f"\n✓ Batch loaded successfully!")
        print(f"\nBatch structure:")
        print(f"  obs keys: {list(batch['obs'].keys())}")
        for key, value in batch['obs'].items():
            if torch.is_tensor(value):
                print(f"    - {key}: {value.shape}")
            else:
                print(f"    - {key}: {type(value)}")
        
        print(f"\n  action: {batch['action'].shape}")
        print(f"  action_is_pad: {batch['action_is_pad'].shape}")
        
        print(f"\n✓ Test 3 PASSED!")
        return True
        
    except Exception as e:
        print(f"\n✗ Test 3 FAILED!")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_val_split():
    """Test 4: Validation split"""
    print("\n" + "=" * 70)
    print("TEST 4: Validation Split")
    print("=" * 70)
    
    try:
        train_dataset = MjThorToSpocDataset(
            data_path="/weka/prior/datasets/robomolmo/testing_door_14NOV/DoorOpeningDataGenConfig",
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
            split="train",
        )
        
        val_dataset = MjThorToSpocDataset(
            data_path="/weka/prior/datasets/robomolmo/testing_door_14NOV/DoorOpeningDataGenConfig",
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
            split="val",
        )
        
        print(f"✓ Both splits created successfully")
        print(f"\nTrain dataset:")
        print(f"  - Data path: {train_dataset.data_path}")
        print(f"  - Use split subdirs: {train_dataset.use_split_subdirs}")
        print(f"  - Files found: {len(train_dataset._files)}")
        print(f"  - Total trajectories: {len(train_dataset.traj_indices)}")
        print(f"  - Train indices: {len(train_dataset.train_indices)}")
        print(f"  - Val indices: {len(train_dataset.val_indices)}")
        print(f"  - is_train: {train_dataset.is_train}")
        print(f"  - Total samples: {len(train_dataset)}")
        
        print(f"\nVal dataset:")
        print(f"  - Data path: {val_dataset.data_path}")
        print(f"  - Use split subdirs: {val_dataset.use_split_subdirs}")
        print(f"  - Files found: {len(val_dataset._files)}")
        print(f"  - Total trajectories: {len(val_dataset.traj_indices)}")
        print(f"  - Train indices: {len(val_dataset.train_indices)}")
        print(f"  - Val indices: {len(val_dataset.val_indices)}")
        print(f"  - is_train: {val_dataset.is_train}")
        print(f"  - Total samples: {len(val_dataset)}")
        
        if len(train_dataset) > 0 and len(val_dataset) > 0:
            print(f"\n✓ Loading one sample from each split...")
            train_sample = train_dataset[0]
            val_sample = val_dataset[0]
            
            print(f"✓ Samples loaded successfully!")
            print(f"  Train sample action shape: {train_sample['action'].shape}")
            print(f"  Val sample action shape: {val_sample['action'].shape}")
            
            print(f"\n✓ Test 4 PASSED!")
            return True
        elif len(train_dataset) == 0:
            print(f"\n✗ Train dataset is empty!")
            return False
        else:
            print(f"\n✗ Val dataset is empty!")
            return False
            
    except Exception as e:
        print(f"\n✗ Test 4 FAILED!")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_normalizer():
    """Test 5: Create normalizer"""
    print("\n" + "=" * 70)
    print("TEST 5: Normalizer Creation")
    print("=" * 70)
    
    try:
        dataset = MjThorToSpocDataset(
            data_path="/weka/prior/datasets/robomolmo/testing_door_14NOV/DoorOpeningDataGenConfig",
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
            split="train",
        )
        
        if len(dataset) == 0:
            print(f"✗ Dataset is empty, skipping normalizer test")
            return False
        
        print(f"✓ Creating normalizer...")
        normalizer = dataset.get_normalizer()
        
        print(f"✓ Normalizer created successfully!")
        print(f"  - Type: {type(normalizer)}")
        print(f"  - Normalizer keys: {list(normalizer.params_dict.keys())}")
        
        # Test normalization on a sample
        sample = dataset[0]
        print(f"\n✓ Testing normalization on sample...")
        
        # Normalize action
        action_normalized = normalizer['action'].normalize(sample['action'])
        action_denormalized = normalizer['action'].unnormalize(action_normalized)
        
        print(f"  - Action shape: {sample['action'].shape}")
        print(f"  - Normalized action range: [{action_normalized.min():.3f}, {action_normalized.max():.3f}]")
        print(f"  - Reconstruction error: {torch.abs(sample['action'] - action_denormalized).max():.6f}")
        
        print(f"\n✓ Test 5 PASSED!")
        return True
        
    except Exception as e:
        print(f"\n✗ Test 4 FAILED!")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "=" * 70)
    print("MjThorToSpoc Dataset Integration Test Suite")
    print("=" * 70)
    print("\nIMPORTANT: Update data_path in this script before running!")
    print("  Current path: /weka/prior/datasets/robomolmo/testing_door_14NOV/DoorOpeningDataGenConfig")
    print("  Dataset will automatically look in 'train' and 'val' subdirectories")
    print("=" * 70)
    
    results = []
    
    # Run tests
    results.append(("Direct Instantiation", test_direct_instantiation()))
    results.append(("Validation Split", test_val_split()))
    results.append(("Hydra Config", test_hydra_config()))
    results.append(("DataLoader", test_dataloader()))
    results.append(("Normalizer", test_normalizer()))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {status}: {name}")
    
    all_passed = all(r[1] for r in results)
    
    print("=" * 70)
    if all_passed:
        print("✓ ALL TESTS PASSED!")
        print("\nYou can now run training with:")
        print("  python ManiFlow/maniflow/workspace/train_mjthor_spoc_workspace.py \\")
        print("      --config-name=maniflow_image_mjthor_spoc")
    else:
        print("✗ SOME TESTS FAILED!")
        print("\nPlease fix the issues above before running training.")
    print("=" * 70)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

