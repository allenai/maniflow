#!/usr/bin/env python3
"""
Simple test script to verify the ManiFlow environment is working.
Can be run inside or outside the Docker container.
"""

import sys
import os

def test_section(name):
    """Print a test section header"""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print('='*60)

def test_result(passed, message):
    """Print test result"""
    symbol = "✓" if passed else "✗"
    print(f"{symbol} {message}")
    return passed

def main():
    all_passed = True
    
    print("\n" + "="*60)
    print("ManiFlow Environment Test")
    print("="*60)
    
    # Test 1: Python version
    test_section("Python Environment")
    python_version = sys.version
    print(f"Python version: {python_version}")
    all_passed &= test_result(True, "Python is available")
    
    # Test 2: PyTorch
    test_section("PyTorch")
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
            print(f"CUDA version: {torch.version.cuda}")
        
        # Test basic operations
        x = torch.randn(3, 3)
        y = torch.matmul(x, x)
        all_passed &= test_result(True, f"PyTorch working (tensor ops: {y.shape})")
    except Exception as e:
        all_passed &= test_result(False, f"PyTorch failed: {e}")
    
    # Test 3: PyTorch3D
    test_section("PyTorch3D")
    try:
        import pytorch3d
        print(f"PyTorch3D version: {pytorch3d.__version__}")
        all_passed &= test_result(True, "PyTorch3D available")
    except Exception as e:
        all_passed &= test_result(False, f"PyTorch3D failed: {e}")
    
    # Test 4: NumPy
    test_section("NumPy")
    try:
        import numpy as np
        print(f"NumPy version: {np.__version__}")
        arr = np.random.randn(3, 3)
        all_passed &= test_result(True, f"NumPy working (array shape: {arr.shape})")
    except Exception as e:
        all_passed &= test_result(False, f"NumPy failed: {e}")
    
    # Test 5: Core dependencies
    test_section("Core Dependencies")
    packages = [
        'scipy',
        'h5py',
        'zarr',
        'wandb',
        'hydra',
        'omegaconf',
        'matplotlib',
    ]
    
    for pkg in packages:
        try:
            __import__(pkg)
            test_result(True, f"{pkg} available")
        except ImportError:
            all_passed &= test_result(False, f"{pkg} not available")
    
    # Test 6: Robotwin dependencies
    test_section("Robotwin Dependencies")
    try:
        import mplib
        print(f"mplib version: {mplib.__version__}")
        all_passed &= test_result(True, "mplib available")
    except Exception as e:
        all_passed &= test_result(False, f"mplib failed: {e}")
    
    try:
        import sapien
        print(f"sapien version: {sapien.__version__}")
        all_passed &= test_result(True, "sapien available")
    except Exception as e:
        all_passed &= test_result(False, f"sapien failed: {e}")
    
    # Test 7: MuJoCo environment
    test_section("MuJoCo Environment")
    mujoco_gl = os.getenv('MUJOCO_GL')
    ld_lib_path = os.getenv('LD_LIBRARY_PATH', '')
    pythonpath = os.getenv('PYTHONPATH', '')
    
    print(f"MUJOCO_GL: {mujoco_gl}")
    print(f"MuJoCo in LD_LIBRARY_PATH: {'/mujoco210/bin' in ld_lib_path}")
    print(f"PYTHONPATH: {pythonpath}")
    
    all_passed &= test_result(
        mujoco_gl == 'egl' and '/mujoco210/bin' in ld_lib_path,
        "MuJoCo environment configured"
    )
    
    # Test 8: Third-party packages (if code is mounted)
    test_section("Third-Party Packages (if mounted)")
    third_party_packages = [
        ('gym', 'Gym'),
        ('metaworld', 'Metaworld'),
    ]
    
    for module, name in third_party_packages:
        try:
            __import__(module)
            test_result(True, f"{name} available")
        except ImportError:
            test_result(False, f"{name} not available (OK if code not mounted)")
    
    # Test 9: ManiFlow (if code is mounted)
    test_section("ManiFlow Package (if mounted)")
    try:
        import maniflow
        all_passed &= test_result(True, "ManiFlow available")
        print(f"ManiFlow location: {maniflow.__file__}")
    except ImportError as e:
        test_result(False, f"ManiFlow not available: {e}")
        print("  (This is expected if source code is not mounted)")
    
    # Test 10: File paths
    test_section("File System")
    paths_to_check = [
        '/root/.mujoco/mujoco210',
        '/root/ManiFlow_Policy',
    ]
    
    for path in paths_to_check:
        exists = os.path.exists(path)
        test_result(exists, f"{path} {'exists' if exists else 'not found'}")
    
    # Final summary
    print("\n" + "="*60)
    if all_passed:
        print("✓ All critical tests passed!")
        print("="*60)
        print("\nEnvironment is ready for use.")
        return 0
    else:
        print("✗ Some tests failed")
        print("="*60)
        print("\nSome issues detected. Check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

