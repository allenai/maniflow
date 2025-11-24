FROM ghcr.io/allenai/cuda:12.8-dev-ubuntu22.04-torch2.7.1-v1.2.199 AS conda_env_builder

ENV APP_HOME=/root/ManiFlow_Policy
WORKDIR $APP_HOME

# Install Vulkan and other system dependencies
RUN apt-get update && \
    apt-get install -y \
    libvulkan1 \
    mesa-vulkan-drivers \
    vulkan-tools \
    git-lfs \
    ninja-build \
    wget \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    patchelf \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create conda environment with Python 3.10
RUN /opt/miniconda3/bin/conda create -n maniflow -y python=3.10

RUN /opt/miniconda3/bin/conda install -n maniflow -y -c conda-forge setuptools wheel ninja
RUN /opt/miniconda3/bin/conda clean -ya

FROM conda_env_builder AS mujoco_installer

ENV PYTHON=/opt/miniconda3/envs/maniflow/bin/python
ENV PIP=/opt/miniconda3/envs/maniflow/bin/pip

# Install MuJoCo 2.1.0 in ~/.mujoco
RUN mkdir -p /root/.mujoco && \
    cd /root/.mujoco && \
    wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz -O mujoco210.tar.gz --no-check-certificate && \
    tar -xvzf mujoco210.tar.gz && \
    rm mujoco210.tar.gz

# Set MuJoCo environment variables
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/root/.mujoco/mujoco210/bin
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/lib/nvidia
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/lib64
ENV MUJOCO_GL=egl
ENV PYOPENGL_PLATFORM=egl
ENV MUJOCO_EGL_DEVICE_ID=0
ENV CUDA_VISIBLE_DEVICES=0

FROM mujoco_installer AS requirements_installer

# Copy requirements and setup files first for better layer caching
COPY scripts/requirements.txt $APP_HOME/scripts/requirements.txt

# Install torch first in the conda environment (needed for PyTorch3D build)
RUN $PIP install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install Python requirements (excluding packages that conflict with base image or need special handling)
# - torch/torchvision: already installed above
# - mplib/sapien: will be installed with specific versions for robotwin compatibility later
# - numpy: requirements.txt has 1.23.5 but numba 0.61.2 needs >=1.24, so relax to compatible version
RUN grep -v "^torch==" $APP_HOME/scripts/requirements.txt | \
    grep -v "^torchvision" | \
    grep -v "^mplib==" | \
    grep -v "^sapien==" | \
    grep -v "^numpy==" > /tmp/filtered_requirements.txt && \
    $PIP install --no-cache-dir "numpy>=1.24,<2.0" && \
    $PIP install --no-cache-dir -r /tmp/filtered_requirements.txt

# Install PyTorch3D (requires torch to be installed first, use --no-build-isolation to access torch)
RUN $PIP install --no-cache-dir --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@stable"

# Install flash attention (optional but useful)
RUN MAX_JOBS=4 $PIP install --no-cache-dir -v flash-attn --no-build-isolation || \
    echo "Flash attention installation failed - continuing without it"

FROM requirements_installer AS dependency_installer

# Note: ManiFlow and third_party source code are NOT copied into the image
# They will be mounted at runtime at $APP_HOME
# This keeps the image smaller and allows for easier development iteration

# Install robotwin dependencies and modify mplib
RUN $PIP install --no-cache-dir mplib==0.1.1 sapien==3.0.0b1

# Modify mplib planner.py to fix robotwin compatibility issues
RUN MPLIB_LOCATION=$($PIP show mplib | grep 'Location' | awk '{print $2}')/mplib && \
    PLANNER=$MPLIB_LOCATION/planner.py && \
    sed -i -E 's/^(\s*)(.*convex=True.*)/\1# \2/' $PLANNER && \
    sed -i -E 's/(if np\.linalg\.norm\(delta_twist\) < 1e-4 )(or collide )(or not within_joint_limit:)/\1\3/g' $PLANNER

# Verify core dependencies installation (source code packages will be available at runtime)
RUN CUDA_VISIBLE_DEVICES="" $PYTHON -c "\
import sys; \
print('=== Verifying Python Environment ==='); \
print('Python version:', sys.version); \
print('Python executable:', sys.executable); \
print(); \
print('=== Testing PyTorch ==='); \
import torch; \
print('PyTorch version:', torch.__version__); \
print('CUDA available:', torch.cuda.is_available()); \
print('CUDA version:', torch.version.cuda); \
x = torch.randn(3, 3); \
y = torch.matmul(x, x); \
print('PyTorch tensor operations working:', y.shape); \
print(); \
print('=== Testing PyTorch3D ==='); \
import pytorch3d; \
print('PyTorch3D version:', pytorch3d.__version__); \
print(); \
print('=== Testing robotwin dependencies ==='); \
import mplib; \
import sapien; \
print('mplib version:', mplib.__version__); \
print('sapien version:', sapien.__version__); \
print(); \
print('=== Core Dependencies Verification Complete ==='); \
print('Note: ManiFlow and third-party packages will be available when source is mounted at runtime'); \
print('All pre-installed packages verified - ready for use!');"

FROM dependency_installer AS final

# Set working directory back to app home
WORKDIR $APP_HOME

# Note: Source code will be mounted at runtime at $APP_HOME
# Set PYTHONPATH to include mounted code locations
ENV PYTHONPATH=${APP_HOME}:${APP_HOME}/ManiFlow:${APP_HOME}/third_party:${PYTHONPATH:-}

# Aggressive cleanup to reduce image size
RUN /opt/miniconda3/bin/conda clean -ya && \
    $PIP cache purge && \
    find /opt/miniconda3 -type f -name "*.pyc" -delete && \
    find /opt/miniconda3 -type d -name "__pycache__" -delete && \
    rm -rf /opt/miniconda3/pkgs/* && \
    rm -rf /root/.cache/pip && \
    rm -rf /tmp/* /var/tmp/* && \
    touch /root/.git-credentials

# Note: Source code is NOT copied into image - mount it at runtime at $APP_HOME
# Usage: docker run --gpus all -it --rm -v $(pwd):${APP_HOME} maniflow-policy:latest
#
# Optional: Download and place assets in your local directory before mounting:
# - dexart assets: https://drive.google.com/file/d/1DxRfB4087PeM3Aejd6cR-RQVgOKdNrL4/view?usp=sharing
#   Place in: ./third_party/dexart-release/assets
# - Adroit RL experts: https://1drv.ms/u/s!Ag5QsBIFtRnTlFWqYWtS2wMMPKNX?e=dw8hsS
#   Place in: ./third_party/VRL3/ckpts
# - robotwin models: https://drive.google.com/file/d/1VOvXZMWQU8-Y1-T2Si5SQLxdH6Eh8nVm/view?usp=sharing
#   Place in: ./ManiFlow/maniflow/env/robotwin
# - robotwin assets: https://drive.google.com/file/d/1VPyzWJYNxQUMf3KSObCyjhawIZMPZExM/view?usp=sharing
#   Place in: ./ManiFlow/maniflow/env/robotwin

# The -l flag makes bash act as a login shell and load /etc/profile, etc.
ENTRYPOINT ["bash", "-l"]

