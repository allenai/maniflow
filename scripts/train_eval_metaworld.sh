
# Usage examples:
# bash scripts/train_eval_metaworld.sh maniflow_pointcloud_policy_metaworld metaworld_multitask_mp debug 0 1_2_3
# bash scripts/train_eval_metaworld.sh maniflow_pointcloud_policy_metaworld metaworld_multitask debug 0 1

DEBUG=False
save_ckpt=True
train=True
eval=True # only set eval=True when save_ckpt=True

alg_name=${1}
task_name=${2}
addition_info=${3}
seed=${4}
gpu_id=${5}
eval_env_processes=8  # Number of parallel processes for evaluation in total, adjust based on your GPU memory


# Setup paths and configuration
base_path="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dataset_path=${base_path}/data
exp_name=${task_name}-${alg_name}-${addition_info}
run_dir="${base_path}/data/outputs/${exp_name}_seed${seed}"
config_name=${alg_name}


# Environment setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export VK_ICD_FILENAMES="${SCRIPT_DIR}/nvidia_icd.json"
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=$(echo $gpu_id | tr '_' ',')

# env_device setup for multi-gpu eval
IFS='_' read -ra gpu_id_arr <<< "$gpu_id"
num_gpus=${#gpu_id_arr[@]}
env_device=""
for i in $(echo $gpu_id | tr "_" "\n")
do
    env_device+="\"cuda:${i}\","
done
env_device=[${env_device%,}]  # Remove trailing comma
echo -e "\033[33mEnvironment devices: ${env_device}\033[0m"



# 27 simple tasks + 11 medium tasks + 10 hard tasks --> 48 tasks in total
task_list=(
    "button-press"
    "button-press-topdown"
    "button-press-topdown-wall"
    "button-press-wall"
    "coffee-button"
    "dial-turn"
    "door-close"
    "door-lock"
    "door-open"
    "door-unlock"
    "drawer-close"
    "drawer-open"
    "faucet-close"
    "faucet-open"
    "handle-press"
    "handle-pull"
    "handle-pull-side"
    "lever-pull"
    "plate-slide"
    "plate-slide-back"
    "plate-slide-back-side"
    "plate-slide-side"
    "reach"
    "reach-wall"
    "window-close"
    "window-open"
    "peg-unplug-side"
    "basketball"
    "bin-picking"
    "box-close"
    "coffee-pull"
    "coffee-push"
    "hammer"
    "peg-insert-side"
    "push-wall"
    "soccer"
    "sweep"
    "sweep-into"
    "assembly"
    "hand-insert"
    "pick-out-of-hole"
    "pick-place"
    "push"
    "shelf-place"
    "disassemble"
    "stick-pull"
    "stick-push"
    "pick-place-wall"
)
task_list_str=$(IFS=,; echo "${task_list[*]}")

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
    python train_maniflow_metaworld_multitask_ddp_workspace.py \
        --config-name=${config_name}.yaml \
        training.num_gpus=${num_gpus} \
        training.distributed=True \
        task=${task_name} \
        task.dataset.data_path=${dataset_path} \
        task.task_names=[${task_list_str}] \
        hydra.run.dir=${run_dir} \
        training.debug=$DEBUG \
        training.seed=${seed} \
        training.device="cuda:0" \
        training.env_device=${env_device} \
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
    python eval_maniflow_metaworld_workspace.py \
        --config-name=${config_name}.yaml \
        task=${task_name} \
        task.dataset.data_path=${zarr_path} \
        task.task_names=[${task_list_str}] \
        task.env_runner.max_processes=${eval_env_processes} \
        hydra.run.dir=${run_dir} \
        training.debug=$DEBUG \
        training.seed=${seed} \
        training.device="cuda:0" \
        training.env_device=${env_device} \
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