
# Usage examples:
# bash scripts/train_eval_robotwin.sh maniflow_pointcloud_policy_robotwin pick_apple_messy_pointcloud 0825 0 1
# bash scripts/train_eval_robotwin.sh maniflow_pointcloud_policy_robotwin diverse_bottles_pick_pointcloud 0825 1 1
# bash scripts/train_eval_robotwin.sh maniflow_image_timm_policy_robotwin pick_apple_messy_image image_debug 0 1
# bash scripts/train_eval_robotwin.sh maniflow_image_transformer_policy_robotwin pick_apple_messy_image image_debug 0 1


DEBUG=False
save_ckpt=True
train=True
eval=True

alg_name=${1}
task_name=${2}
addition_info=${3}
seed=${4}
gpu_id=${5}

# Training/Evaluation parameters
eval_episode=100
eval_mode="latest"  # "best" or "latest"
num_inference_steps=10
n_obs_steps=2
horizon=16
n_action_steps=16

# Validate required arguments
if [[ -z "$alg_name" || -z "$task_name" || -z "$addition_info" || -z "$seed" || -z "$gpu_id" ]]; then
    echo "Usage: $0 <alg_name> <task_name> <addition_info> <seed> <gpu_id>"
    echo "Example: $0 maniflow_pointcloud_policy_robotwin pick_apple_messy_pointcloud pointcloud_debug 0 1"
    exit 1
fi

# Process task name (remove _image or _pointcloud suffix)
processed_task_name=${task_name}
if [[ $task_name == *"_image"* ]]; then
    processed_task_name=${task_name//_image/}
elif [[ $task_name == *"_pointcloud"* ]]; then
    processed_task_name=${task_name//_pointcloud/}
fi

# Setup paths and configuration
base_path="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_zarr_name=${processed_task_name}
zarr_path="${base_path}/data/${task_zarr_name}.zarr"
exp_name=${processed_task_name}-${alg_name}-${addition_info}
# run_dir="/gscratch/scrubbed/geyan/projects/ManiFlow_Policy/ManiFlow/data/outputs/${exp_name}_seed${seed}"
run_dir="${base_path}/data/outputs/${exp_name}_seed${seed}"
config_name=${alg_name}


# Environment setup
# Environment setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export VK_ICD_FILENAMES="${SCRIPT_DIR}/nvidia_icd.json"
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${gpu_id}


# Set wandb mode based on debug flag
if [ $DEBUG = True ]; then
    wandb_mode=offline
    echo -e "\033[33m=== DEBUG MODE ===\033[0m"
else
    wandb_mode=online
    echo -e "\033[33m=== TRAINING MODE ===\033[0m"
fi


# Print configuration
echo -e "\033[33mTask: ${processed_task_name}\033[0m"
echo -e "\033[33mGPU ID: ${gpu_id}\033[0m"
echo -e "\033[33mTrain: ${train}, Eval: ${eval}\033[0m"


# Change to workspace directory
cd maniflow/workspace

# Training phase
if [ $train = True ]; then
    echo -e "\033[32m=== Starting Training ===\033[0m"
    python train_maniflow_robotwin_workspace.py \
        --config-name=${config_name}.yaml \
        robotwin_task=${task_name} \
        robotwin_task.dataset.zarr_path=${zarr_path} \
        hydra.run.dir=${run_dir} \
        training.debug=$DEBUG \
        training.seed=${seed} \
        training.device="cuda:0" \
        exp_name=${exp_name} \
        logging.mode=${wandb_mode} \
        checkpoint.save_ckpt=${save_ckpt}
    
    if [ $? -eq 0 ]; then
        echo -e "\033[32m=== Training completed successfully ===\033[0m"
    else
        echo -e "\033[31m=== Training failed ===\033[0m"
        exit 1
    fi
else
    echo -e "\033[33m=== Skipping Training ===\033[0m"
fi

# Evaluation phase
if [ $eval = True ]; then
    echo -e "\033[32m=== Starting Evaluation ===\033[0m"
    python eval_maniflow_robotwin_workspace.py \
        --config-name=${config_name}.yaml \
        +eval_mode=${eval_mode} \
        robotwin_task=${task_name} \
        robotwin_task.env_runner.eval_episodes=${eval_episode} \
        hydra.run.dir=${run_dir} \
        training.debug=$DEBUG \
        training.seed=${seed} \
        training.device="cuda:0" \
        policy.num_inference_steps=${num_inference_steps} \
        policy.n_obs_steps=${n_obs_steps} \
        policy.horizon=${horizon} \
        policy.n_action_steps=${n_action_steps} \
        exp_name=${exp_name} \
        logging.mode=${wandb_mode} \
        checkpoint.save_ckpt=${save_ckpt}
    
    if [ $? -eq 0 ]; then
        echo -e "\033[32m=== Evaluation completed successfully ===\033[0m"
    else
        echo -e "\033[31m=== Evaluation failed ===\033[0m"
        exit 1
    fi
else
    echo -e "\033[33m=== Skipping Evaluation ===\033[0m"
fi

echo -e "\033[32m=== Script completed ===\033[0m"