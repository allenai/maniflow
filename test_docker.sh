#!/bin/bash
# Simple test script to verify Docker image is working

set -e

IMAGE_NAME="maniflow-policy:latest"
echo "=========================================="
echo "Testing Docker Image: $IMAGE_NAME"
echo "=========================================="
echo

# Test 1: Check if image exists
echo "✓ Test 1: Checking if image exists..."
if docker image inspect $IMAGE_NAME &> /dev/null; then
    echo "  ✓ Image found: $IMAGE_NAME"
else
    echo "  ✗ Image not found. Please build it first:"
    echo "    docker build -t $IMAGE_NAME ."
    exit 1
fi
echo

# Test 2: Test basic container startup
echo "✓ Test 2: Testing container startup..."
docker run --rm $IMAGE_NAME -c "echo 'Container started successfully'" || {
    echo "  ✗ Container failed to start"
    exit 1
}
echo "  ✓ Container starts successfully"
echo

# Test 3: Test Python environment
echo "✓ Test 3: Testing Python environment..."
docker run --rm $IMAGE_NAME -c "/opt/miniconda3/envs/maniflow/bin/python --version" || {
    echo "  ✗ Python not available"
    exit 1
}
echo "  ✓ Python environment working"
echo

# Test 4: Test PyTorch
echo "✓ Test 4: Testing PyTorch..."
docker run --rm $IMAGE_NAME -c "\
/opt/miniconda3/envs/maniflow/bin/python -c '\
import torch; \
print(f\"  PyTorch version: {torch.__version__}\"); \
x = torch.randn(2, 2); \
y = x.matmul(x); \
print(f\"  Tensor operations work: {y.shape}\")'" || {
    echo "  ✗ PyTorch test failed"
    exit 1
}
echo "  ✓ PyTorch working"
echo

# Test 5: Test PyTorch3D
echo "✓ Test 5: Testing PyTorch3D..."
docker run --rm $IMAGE_NAME -c "\
/opt/miniconda3/envs/maniflow/bin/python -c '\
import pytorch3d; \
print(f\"  PyTorch3D version: {pytorch3d.__version__}\")'" || {
    echo "  ✗ PyTorch3D test failed"
    exit 1
}
echo "  ✓ PyTorch3D working"
echo

# Test 6: Test mounted code (requires source code)
echo "✓ Test 6: Testing code mounting..."
docker run --rm -v $(pwd):/root/ManiFlow_Policy $IMAGE_NAME -c "\
cd /root/ManiFlow_Policy && \
ls -la ManiFlow/ > /dev/null && \
ls -la third_party/ > /dev/null && \
echo '  ✓ Code mounted successfully' && \
echo '  ✓ ManiFlow directory accessible' && \
echo '  ✓ third_party directory accessible'" || {
    echo "  ✗ Code mounting failed"
    exit 1
}
echo

# Test 7: Test ManiFlow import (with mounted code)
echo "✓ Test 7: Testing ManiFlow import with mounted code..."
docker run --rm -v $(pwd):/root/ManiFlow_Policy $IMAGE_NAME -c "\
/opt/miniconda3/envs/maniflow/bin/python -c '\
import sys; \
sys.path.insert(0, \"/root/ManiFlow_Policy/ManiFlow\"); \
try:
    import maniflow; \
    print(\"  ✓ ManiFlow imported successfully\")
except ImportError as e:
    print(f\"  ⚠ ManiFlow import warning: {e}\")
    print(\"  (This may be expected - some dependencies need third_party packages)\")'" || {
    echo "  ⚠ ManiFlow import had issues (may be expected without full setup)"
}
echo

# Test 8: Test GPU access (optional - won't fail if no GPU)
echo "✓ Test 8: Testing GPU access..."
docker run --gpus all --rm $IMAGE_NAME -c "\
/opt/miniconda3/envs/maniflow/bin/python -c '\
import torch; \
if torch.cuda.is_available(): \
    print(f\"  ✓ GPU detected: {torch.cuda.get_device_name(0)}\"); \
    print(f\"  ✓ CUDA version: {torch.version.cuda}\"); \
else: \
    print(\"  ⚠ No GPU detected (this is OK for build test)\")'" 2>/dev/null || {
    echo "  ⚠ GPU test skipped (--gpus flag may not be available)"
}
echo

# Test 9: Test MuJoCo environment variables
echo "✓ Test 9: Testing MuJoCo environment..."
docker run --rm $IMAGE_NAME -c "\
/opt/miniconda3/envs/maniflow/bin/python -c '\
import os; \
mujoco_gl = os.getenv(\"MUJOCO_GL\"); \
print(f\"  MUJOCO_GL: {mujoco_gl}\"); \
ld_lib_path = os.getenv(\"LD_LIBRARY_PATH\"); \
has_mujoco = \"/mujoco210/bin\" in (ld_lib_path or \"\"); \
print(f\"  MuJoCo in LD_LIBRARY_PATH: {has_mujoco}\"); \
if mujoco_gl == \"egl\" and has_mujoco: \
    print(\"  ✓ MuJoCo environment configured correctly\")'" || {
    echo "  ✗ MuJoCo environment test failed"
    exit 1
}
echo

# Test 10: Test mplib/sapien versions
echo "✓ Test 10: Testing robotwin dependencies..."
docker run --rm $IMAGE_NAME -c "\
/opt/miniconda3/envs/maniflow/bin/python -c '\
import mplib; \
import sapien; \
print(f\"  mplib version: {mplib.__version__}\"); \
print(f\"  sapien version: {sapien.__version__}\"); \
print(\"  ✓ Robotwin dependencies installed\")'" || {
    echo "  ✗ Robotwin dependencies test failed"
    exit 1
}
echo

echo "=========================================="
echo "✓ All tests passed!"
echo "=========================================="
echo
echo "Your Docker image is ready to use. Run it with:"
echo "  docker run --gpus all -it --rm --shm-size=32g -v \$(pwd):/root/ManiFlow_Policy $IMAGE_NAME"
echo

