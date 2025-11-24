
FROM ghcr.io/allenai/cuda:12.8-dev-ubuntu22.04-torch2.7.1-v1.2.199 AS conda_env_builder


ENV APP_HOME /root/mujoco-thor
WORKDIR $APP_HOME

# Update package lists and install git-lfs, build tools, wget, and FFmpeg (required for decord)
RUN apt-get update && \
    apt-get install -y git-lfs ninja-build wget ffmpeg libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libswresample-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN /opt/miniconda3/bin/conda create -n mjthor -y python=3.10

RUN /opt/miniconda3/bin/conda install -n mjthor -y -c conda-forge setuptools wheel ninja
RUN /opt/miniconda3/bin/conda clean -ya

FROM conda_env_builder AS requirements_installer

ARG GITHUB_TOKEN

ENV PYTHON=/opt/miniconda3/envs/mjthor/bin/python
ENV PIP=/opt/miniconda3/envs/mjthor/bin/pip

# CUDA development tools are already installed in base image
# Verify they're available
RUN echo "=== Verifying CUDA from base image ===" && \
    nvcc --version

# Set environment variables for cuRobo compilation with multi-GPU support
# Set CUDA architecture for Quadro RTX 8000, A6000, L40, A100, H100, and B200
ENV TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0"
ENV CUDA_VISIBLE_DEVICES=0
ENV CUDA_LAUNCH_BLOCKING=1

# MuJoCo headless rendering environment variables
ENV MUJOCO_GL=egl
ENV PYOPENGL_PLATFORM=egl
ENV MUJOCO_EGL_DEVICE_ID=0


# Copy only pyproject.toml first for better layer caching
COPY ./pyproject.toml $APP_HOME/pyproject.toml

# Install mujoco and mujoco-mjx from wheels/git BEFORE installing the project
# This prevents pip from trying to install mujoco==3.3.8 from PyPI (which doesn't exist)
# Install mujoco 3.3.8 directly from wheel URL (pip can install from URLs)
RUN $PIP install --no-cache-dir https://py.mujoco.org/mujoco/mujoco-3.3.8.dev832359379-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
# Install mujoco-mjx 3.3.8 from git (not available on PyPI yet)
RUN $PIP install --no-cache-dir "git+https://github.com/google-deepmind/mujoco@4e46db89037de9a2e388dfbb830b97ec37c4326c#subdirectory=mjx"

# Install project with --no-deps to prevent pip from trying to resolve mujoco==3.3.8 from PyPI
RUN ( \
    export PIP_SRC=/opt/miniconda3/envs/mjthor/pipsrc; \
    cd $APP_HOME \
    && $PIP install --no-cache-dir -e . \
    && $PIP install --no-cache-dir --upgrade "typing-extensions>=4.14.1" \
    && $PIP install --no-cache-dir --no-build-isolation -e git+https://x-access-token:${GITHUB_TOKEN}@github.com/allenai/curobo.git@417c995647fcb173a2bc094d1284b2a4f4b000ad#egg=nvidia-curobo \
    && $PIP install --no-cache-dir --no-build-isolation --no-deps -e git+https://x-access-token:${GITHUB_TOKEN}@github.com/allenai/mujoco-thor-resources.git@e6636f8ad9f5a06a678104dbf598bde5546339d1#egg=mujoco_thor_resources \
    && $PIP cache purge \
)

# Verify all critical dependencies are installed correctly
RUN CUDA_VISIBLE_DEVICES="" $PYTHON -c \
    "import sys; \
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
    print('=== Testing MuJoCo ==='); \
    import mujoco; \
    print('MuJoCo version:', mujoco.__version__); \
    import mujoco.mjx as mjx; \
    print('MuJoCo-MJX imported successfully'); \
    print(); \
    print('=== Testing decord ==='); \
    import decord; \
    print('decord imported successfully'); \
    from decord import VideoReader, cpu; \
    print('decord VideoReader imported successfully'); \
    print(); \
    print('=== Testing cuRobo ==='); \
    import curobo; \
    print('cuRobo imported successfully!'); \
    from curobo.types.math import Pose; \
    print('cuRobo core types imported successfully!'); \
    print(); \
    print('=== Testing mujoco-thor dependencies ==='); \
    import numpy as np; \
    import jax; \
    import gymnasium; \
    import h5py; \
    import imageio; \
    import scipy; \
    print('All core dependencies imported successfully!'); \
    print(); \
    print('=== Installation Verification Complete ==='); \
    print('All packages installed and verified - ready for use!')"


FROM requirements_installer AS final

# Set working directory back to app home
WORKDIR $APP_HOME

# Note: Source code is NOT copied into image - mount it at runtime at $APP_HOME
# This keeps the image smaller and more reusable

# Set default MuJoCo-THOR assets directory (can be overridden at runtime)
ENV WEKA_DEFAULT_MJCTHOR_ASSETS_DIR=/weka/prior/datasets/robomolmo/mjthor_resources/

# MuJoCo-THOR resources configuration
ENV MJCTHOR_ASSETS_DIR=/root/assets
ENV MJCTHOR_AUTO_INSTALL=True

# Set PYTHONPATH to include mounted code directory
ENV PYTHONPATH=$APP_HOME:$PYTHONPATH

# Aggressive cleanup to reduce image size
RUN /opt/miniconda3/bin/conda clean -ya \
    && $PIP cache purge \
    && find /opt/miniconda3 -type f -name "*.pyc" -delete \
    && find /opt/miniconda3 -type d -name "__pycache__" -delete \
    && rm -rf /opt/miniconda3/pkgs/* \
    && rm -rf /root/.cache/pip \
    && rm -rf /tmp/* /var/tmp/* \
    && touch /root/.git-credentials

# The -l flag makes bash act as a login shell and load /etc/profile, etc.
ENTRYPOINT ["bash", "-l"]
