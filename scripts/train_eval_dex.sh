
# Usage examples:
# bash scripts/train_eval_dex.sh maniflow_pointcloud_policy_dex adroit_door_pointcloud pointcloud_debug 0 1
# bash scripts/train_eval_dex.sh maniflow_image_timm_policy_dex adroit_door_image image_debug 0 1
# bash scripts/train_eval_dex.sh maniflow_pointcloud_policy_dex dexart_laptop_pointcloud pointcloud_debug 0 1
# bash scripts/train_eval_dex.sh maniflow_image_timm_policy_dex dexart_laptop_image image_debug 0 1

DEBUG=False
save_ckpt=False
train=True
eval=False # set to false since mostly we do online eval for adroit/dexart tasks

alg_name=${1}
task_name=${2}
addition_info=${3}
seed=${4}
gpu_id=${5}

# Process task name (remove _image or _pointcloud suffix)
processed_task_name=${task_name}
if [[ $task_name == *"_image"* ]]; then
    processed_task_name=${task_name//_image/}
elif [[ $task_name == *"_pointcloud"* ]]; then
    processed_task_name=${task_name//_pointcloud/}
fi

# Example dataset paths:
# dataset_path=data/adroit_hammer_expert.zarr
# dataset_path=data/adroit_door_expert.zarr
# dataset_path=data/adroit_pen_expert.zarr

# dataset_path=data/dexart_faucet_expert.zarr
# dataset_path=data/dexart_laptop_expert.zarr
# dataset_path=data/dexart_bucket_expert.zarr
# dataset_path=data/dexart_toilet_expert.zarr

# dataset_path=data/metaworld_basketball_expert.zarr
# dataset_path=data/metaworld_assembly_expert.zarr


# Setup paths and configuration
base_path="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dataset_path=data/${processed_task_name}_expert.zarr # e.g., data/adroit_door_expert.zarr, data/dexart_faucet_expert.zarr, data/metaworld_basketball_expert.zarr
zarr_path="${base_path}/${dataset_path}"
exp_name=${task_name}-${alg_name}-${addition_info}
run_dir="${base_path}/data/outputs/${exp_name}_seed${seed}"
config_name=${alg_name}


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
echo -e "\033[33mTask: ${task_name}\033[0m"
echo -e "\033[33mGPU ID: ${gpu_id}\033[0m"
echo -e "\033[33mTrain: ${train}, Eval: ${eval}\033[0m"


# Change to workspace directory
cd maniflow/workspace

# Training phase
if [ $train = True ]; then
    echo -e "\033[32m=== Starting Training ===\033[0m"
    python train_maniflow_dex_workspace.py \
        --config-name=${config_name}.yaml \
        task=${task_name} \
        task.dataset.zarr_path=${dataset_path} \
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
    python eval_maniflow_dex_workspace.py \
        --config-name=${config_name}.yaml \
        task=${task_name} \
        task.dataset.zarr_path=${zarr_path} \
        hydra.run.dir=${run_dir} \
        training.debug=$DEBUG \
        training.seed=${seed} \
        training.device="cuda:0" \
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