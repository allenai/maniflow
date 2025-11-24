if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

def _copy_to_cpu(x):
    if isinstance(x, torch.Tensor):
        return x.detach().to('cpu')
    elif isinstance(x, dict):
        result = dict()
        for k, v in x.items():
            result[k] = _copy_to_cpu(v)
        return result
    elif isinstance(x, list):
        return [_copy_to_cpu(k) for k in x]
    else:
        return copy.deepcopy(x)

import os
import hydra
import torch
import dill
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import numpy as np
from termcolor import cprint
import shutil
import time
import threading
import sys
sys.path.insert(0, '../')
sys.path.append('maniflow/env_runner')
sys.path.append('maniflow/policy')

from hydra.core.hydra_config import HydraConfig
from maniflow.policy.maniflow_pointcloud_policy import ManiFlowTransformerPointcloudPolicy
from maniflow.dataset.base_dataset import BaseDataset
from maniflow.env_runner.base_runner import BaseRunner
from maniflow.common.checkpoint_util import TopKCheckpointManager
from maniflow.common.pytorch_util import dict_apply, optimizer_to
from maniflow.model.diffusion.ema_model import EMAModel
from maniflow.model.common.lr_scheduler import get_scheduler

OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainManiFlowRoboTwinTransformerWorkspace:
    include_keys = ['global_step', 'epoch']
    exclude_keys = tuple()

    def __init__(self, cfg: OmegaConf, output_dir=None):
        self.cfg = cfg
        self._output_dir = output_dir
        self._saving_thread = None
        
        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: ManiFlowTransformerPointcloudPolicy = hydra.utils.instantiate(cfg.policy)

        self.ema_model: ManiFlowTransformerPointcloudPolicy = None
        if cfg.training.use_ema:
            try:
                self.ema_model = copy.deepcopy(self.model)
            except: # minkowski engine could not be copied. recreate it
                self.ema_model = hydra.utils.instantiate(cfg.policy)


        # configure training state
        # self.optimizer = hydra.utils.instantiate(
        #     cfg.optimizer, params=self.model.parameters())
        self.optimizer = self.model.get_optimizer(**cfg.optimizer)

        # configure training state
        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        WANDB = True
        
        if cfg.training.debug:
            cfg.training.num_epochs = 100
            cfg.training.max_train_steps = 10
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 20
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1
            RUN_ROLLOUT = True
            RUN_CKPT = False
            verbose = True
        else:
            RUN_ROLLOUT = True
            RUN_CKPT = True
            verbose = False
        RUN_ROLLOUT = False
        RUN_VALIDATION = True # reduce time cost
        
        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # configure dataset
        dataset: BaseDataset
        dataset = hydra.utils.instantiate(cfg.robotwin_task.dataset)

        assert isinstance(dataset, BaseDataset), print(f"dataset must be BaseDataset, got {type(dataset)}")
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        # print dataset info
        cprint(f"Dataset: {dataset.__class__.__name__}", 'red')
        cprint(f"Dataset Path: {dataset.zarr_path}", 'red')
        cprint(f"Number of training episodes: {dataset.train_episodes_num}", 'red')
        cprint(f"Number of validation episodes: {dataset.val_episodes_num}", 'red')


        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step-1
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)
        # configure env runner
        env_runner: BaseRunner
        env_runner = hydra.utils.instantiate(
            cfg.robotwin_task.env_runner,
            output_dir=self.output_dir)
        assert isinstance(env_runner, BaseRunner)
        
        cfg.logging.name = str(cfg.robotwin_task.name)
        cprint("-----------------------------", "yellow")
        cprint(f"[WandB] group: {cfg.logging.group}", "yellow")
        cprint(f"[WandB] name: {cfg.logging.name}", "yellow")
        cprint("-----------------------------", "yellow")
        # configure logging
        if WANDB:
            wandb_run = wandb.init(
                dir=str(self.output_dir),
                config=OmegaConf.to_container(cfg, resolve=True),
                **cfg.logging
            )
            wandb.config.update(
                {
                    "output_dir": self.output_dir,
                }
            )

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        # save batch for sampling
        train_sampling_batch = None
        
        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        for local_epoch_idx in range(cfg.training.num_epochs):
            step_log = dict()
            # ========= train for this epoch ==========
            train_losses = list()
            with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                    leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                for batch_idx, batch in enumerate(tepoch):
                    t1 = time.time()
                    # device transfer
                    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                    if train_sampling_batch is None:
                        train_sampling_batch = batch
                
                    # compute loss
                    t1_1 = time.time()
                    
                    # Forward pass
                    raw_loss, loss_dict = self.model.compute_loss(batch, self.ema_model)
                    
                    loss = raw_loss / cfg.training.gradient_accumulate_every
                    loss.backward()
                    
                    t1_2 = time.time()

                    # step optimizer
                    if self.global_step % cfg.training.gradient_accumulate_every == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        lr_scheduler.step()
                    t1_3 = time.time()
                    # update ema
                    if cfg.training.use_ema:
                        ema.step(self.model)
                    t1_4 = time.time()
                    # logging
                    raw_loss_cpu = raw_loss.item()
                    tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                    train_losses.append(raw_loss_cpu)
                    step_log = {
                        'train_loss': raw_loss_cpu,
                        'global_step': self.global_step,
                        'epoch': self.epoch,
                        'lr': lr_scheduler.get_last_lr()[0]
                    }
                    t1_5 = time.time()
                    step_log.update(loss_dict)
                    t2 = time.time()
                    
                    if verbose:
                        print(f"total one step time: {t2-t1:.3f}")
                        print(f" compute loss time: {t1_2-t1_1:.3f}")
                        print(f" step optimizer time: {t1_3-t1_2:.3f}")
                        print(f" update ema time: {t1_4-t1_3:.3f}")
                        print(f" logging time: {t1_5-t1_4:.3f}")

                    is_last_batch = (batch_idx == (len(train_dataloader)-1))
                    if not is_last_batch:
                        # log of last step is combined with validation and rollout
                        if WANDB:
                            wandb_run.log(step_log, step=self.global_step)
                        self.global_step += 1

                    if (cfg.training.max_train_steps is not None) \
                        and batch_idx >= (cfg.training.max_train_steps-1):
                        break

            # at the end of each epoch
            # replace train_loss with epoch average
            train_loss = np.mean(train_losses)
            step_log['train_loss'] = train_loss

            # ========= eval for this epoch ==========
            policy = self.model
            if cfg.training.use_ema:
                policy = self.ema_model
            policy.eval()

            # run rollout
            if cfg.training.debug:
                min_epoch_rollout = 0
            else:
                min_epoch_rollout = 300
            if (self.epoch % cfg.training.rollout_every) == 0 and RUN_ROLLOUT and env_runner is not None and self.epoch >= min_epoch_rollout: # and self.epoch > 1, and self.epoch >= 100
                cprint(f"Running rollout for epoch {self.epoch}", 'cyan')
                t3 = time.time()
                # runner_log = env_runner.run(policy, dataset=dataset)
                runner_log = env_runner.run(policy)
                t4 = time.time()
                # print(f"rollout time: {t4-t3:.3f}")
                # log all
                step_log.update(runner_log)
            elif self.epoch == 0:
                runner_log = dict()
                runner_log['test_mean_score'] = 0
                runner_log['mean_success_rates'] = 0
                runner_log['SR_test_L3'] = 0
                runner_log['SR_test_L5'] = 0
                runner_log['sim_video_eval'] = None
                step_log.update(runner_log)

            # run validation
            if (self.epoch % cfg.training.val_every) == 0 and RUN_VALIDATION:
                with torch.no_grad():
                    val_losses = list()
                    with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
                            leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                        for batch_idx, batch in enumerate(tepoch):
                            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                            
                            # Forward pass
                            loss, loss_dict = self.model.compute_loss(batch, self.ema_model)
                            val_losses.append(loss)
                            print(f'epoch {self.epoch}, eval loss: ', float(loss.cpu()))
                            if (cfg.training.max_val_steps is not None) \
                                and batch_idx >= (cfg.training.max_val_steps-1):
                                break
                    if len(val_losses) > 0:
                        val_loss = torch.mean(torch.tensor(val_losses)).item()
                        # log epoch average validation loss
                        step_log['val_loss'] = val_loss
            
            # run diffusion sampling on a training batch
            if (self.epoch % cfg.training.sample_every) == 0:
                with torch.no_grad():
                    # sample trajectory from training set, and evaluate difference
                    batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                    obs_dict = batch['obs']
                    gt_action = batch['action']
                    
                    result = policy.predict_action(obs_dict)
                    pred_action = result['action_pred']
                    mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                    step_log['train_action_mse_error'] = mse.item()
                    del batch
                    del obs_dict
                    del gt_action
                    del result
                    del pred_action
                    del mse
            
            if env_runner is None or step_log.get('test_mean_score', None) is None:
                step_log['test_mean_score'] = - train_loss

            # checkpoint
            if (self.epoch % cfg.training.checkpoint_every) == 0 and cfg.checkpoint.save_ckpt:

                if cfg.checkpoint.save_last_ckpt:
                    self.save_checkpoint()
                if cfg.checkpoint.save_last_snapshot:
                    self.save_snapshot()

                # sanitize metric names
                metric_dict = dict()
                for key, value in step_log.items():
                    new_key = key.replace('/', '_')
                    metric_dict[new_key] = value
                
                # if not cfg.policy.use_pc_color:
                #     if not os.path.exists(f'checkpoints/{self.cfg.robotwin_task.name}'):
                #         os.makedirs(f'checkpoints/{self.cfg.robotwin_task.name}')
                #     save_path = f'checkpoints/{self.cfg.robotwin_task.name}/{self.epoch + 1}.ckpt'
                # else:
                #     if not os.path.exists(f'checkpoints/{self.cfg.robotwin_task.name}_w_rgb'):
                #         os.makedirs(f'checkpoints/{self.cfg.robotwin_task.name}_w_rgb')
                #     save_path = f'checkpoints/{self.cfg.robotwin_task.name}_w_rgb/{self.epoch + 1}.ckpt'

                # self.save_checkpoint(save_path)
                try:
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)
                except Exception as e:
                    print(f"Error in getting topk ckpt path: {e}")
                    topk_ckpt_path = None

                if topk_ckpt_path is not None:
                    self.save_checkpoint(path=topk_ckpt_path)
                

            # ========= eval end for this epoch ==========
            policy.train()

            # end of epoch
            # log of last step is combined with validation and rollout
            if WANDB:
                wandb_run.log(step_log, step=self.global_step)
            self.global_step += 1
            self.epoch += 1
            del step_log
    
    def eval(self, mode='best'):
        # load the latest checkpoint
        cfg = copy.deepcopy(self.cfg)
        
        lastest_ckpt_path = self.get_checkpoint_path(tag=mode, monitor_key=cfg.checkpoint.topk.monitor_key)
        if lastest_ckpt_path.is_file():
            cprint(f"Resuming from {mode} checkpoint {lastest_ckpt_path}", 'magenta')
            self.load_checkpoint(path=lastest_ckpt_path)
            # print ckpt info
            cprint(f"{self.epoch} epochs, {self.global_step} steps", 'magenta')
        
        # configure env
        env_runner: BaseRunner
        env_runner = hydra.utils.instantiate(
            cfg.robotwin_task.env_runner,
            output_dir=self.output_dir)
        assert isinstance(env_runner, BaseRunner)
        policy = self.model
        if cfg.training.use_ema:
            policy = self.ema_model
        policy.eval()
        policy.cuda()

        # inference_steps = cfg.policy.num_inference_steps
        all_rollout_steps = [10] # [10, 1, 4, 2, 8]
        for inference_steps in all_rollout_steps:
            eval_episodes = cfg.robotwin_task.env_runner.eval_episodes
            cprint(f"Running evaluation for {inference_steps} inference steps", 'magenta')

            horizon = policy.horizon
            n_action_steps = policy.n_action_steps
            cprint(f"Evaluating with horizon={horizon}, n_action_steps={n_action_steps}, eval_episodes={eval_episodes}, inference_steps={inference_steps}", 'magenta')

            # Create eval results directory
            eval_dir = os.path.join(self.output_dir, f'eval_results/{self.epoch}/eval_{eval_episodes}_episodes/horizon{horizon}_act{n_action_steps}/{inference_steps}')
            os.makedirs(eval_dir, exist_ok=True)

            policy.num_inference_steps = inference_steps
            runner_log = env_runner.run(policy)

        
            cprint(f"---------------- Eval Results --------------", 'magenta')
            metrics_dict = {}
            for key, value in runner_log.items():
                if isinstance(value, float):
                    metrics_dict[key] = value
                    cprint(f"{key}: {value:.4f}", 'magenta')
                if isinstance(value, dict):
                    for k, v in value.items():
                        if isinstance(v, float):
                            metrics_dict[f"{key}/{k}"] = v
                            cprint(f"{key}/{k}: {v:.4f}", 'magenta')
            
            # Save metrics to JSON
            import json
            metrics_path = os.path.join(eval_dir, f'metrics_{mode}_{self.epoch}.json')
            with open(metrics_path, 'w') as f:
                json.dump(metrics_dict, f, indent=4)
            
            # Save videos if they exist in runner_log
            runner_log.pop('average_success_rate', None) # Remove average_success_rate from runner_log
            video_id = 0
            task_name = runner_log['task_name']
            for k, v in runner_log.items():
                if 'video' in k:
                    if isinstance(v, np.ndarray):
                        video_dir = os.path.join(eval_dir, 'videos', task_name)
                        os.makedirs(video_dir, exist_ok=True)
                        video_path = os.path.join(video_dir, f'{k}_{mode}_{self.epoch}_{video_id}.mp4')
                        
                        # Convert from N, C, H, W to N, H, W, C format for saving
                        v = np.transpose(v, (0, 2, 3, 1))
                        # Save video using imageio or cv2
                        import imageio
                        imageio.mimsave(video_path, v, fps=10)
                    elif hasattr(v, '_path'):  # Handle wandb.Video object
                        video_dir = os.path.join(eval_dir, 'videos', task_name)
                        os.makedirs(video_dir, exist_ok=True)
                        video_path = os.path.join(video_dir, f'{k}_{mode}_{self.epoch}_{video_id}.mp4')
                        # Copy the video file from wandb path to our eval directory
                        shutil.copy2(v._path, video_path)
                    else:
                        cprint(f"Unknown video format for {k}", 'red')
                    video_id += 1
            cprint(f"Evaluation results saved to {eval_dir}", 'magenta')


    def get_policy_and_runner(self, cfg, checkpoint_num=3000):
        # load the latest checkpoint
        
        cfg = copy.deepcopy(self.cfg)
        env_runner: BaseRunner
        env_runner = hydra.utils.instantiate(
            cfg.robotwin_task.env_runner,
            output_dir=self.output_dir)
        assert isinstance(env_runner, BaseRunner)
        
        if not cfg.policy.use_pc_color:
            ckpt_file = pathlib.Path(f'./checkpoints/{self.cfg.robotwin_task.name}/{checkpoint_num}.ckpt')
        else:
            ckpt_file = pathlib.Path(f'./checkpoints/{self.cfg.robotwin_task.name}_w_rgb/{checkpoint_num}.ckpt')

        print('ckpt file exist:', ckpt_file.is_file())
        
        if ckpt_file.is_file():
            cprint(f"Resuming from checkpoint {ckpt_file}", 'magenta')
            self.load_checkpoint(path=ckpt_file)
        
        policy = self.model
        if cfg.training.use_ema:
            policy = self.ema_model
    
        policy.eval()
        policy.cuda()
        return policy, env_runner

    @property
    def output_dir(self):
        output_dir = self._output_dir
        if output_dir is None:
            output_dir = HydraConfig.get().runtime.output_dir
        return output_dir
    

    def save_checkpoint(self, path=None, tag='latest', 
            exclude_keys=None,
            include_keys=None,
            use_thread=False):
        print('saved in ', path)
        if path is None:
            path = pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')
        else:
            path = pathlib.Path(path)
        if exclude_keys is None:
            exclude_keys = tuple(self.exclude_keys)
        if include_keys is None:
            include_keys = tuple(self.include_keys) + ('_output_dir',)
            
        path.parent.mkdir(parents=False, exist_ok=True)
        payload = {
            'cfg': self.cfg,
            'state_dicts': dict(),
            'pickles': dict()
        } 

        for key, value in self.__dict__.items():
            if hasattr(value, 'state_dict') and hasattr(value, 'load_state_dict'):
                # modules, optimizers and samplers etc
                if key not in exclude_keys:
                    if use_thread:
                        payload['state_dicts'][key] = _copy_to_cpu(value.state_dict())
                    else:
                        payload['state_dicts'][key] = value.state_dict()
            elif key in include_keys:
                payload['pickles'][key] = dill.dumps(value)
        if use_thread:
            self._saving_thread = threading.Thread(
                target=lambda : torch.save(payload, path.open('wb'), pickle_module=dill))
            self._saving_thread.start()
        else:
            torch.save(payload, path.open('wb'), pickle_module=dill)
        
        del payload
        torch.cuda.empty_cache()
        return str(path.absolute())
    
    def get_checkpoint_path(self, tag='latest', monitor_key='test_mean_score'):
        if tag=='latest':
            return pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')
        elif tag=='best': 
            # the checkpoints are saved as format: epoch={}-test_mean_score={}.ckpt
            # find the best checkpoint
            checkpoint_dir = pathlib.Path(self.output_dir).joinpath('checkpoints')
            all_checkpoints = os.listdir(checkpoint_dir)
            best_ckpt = None
            best_score = -1e10 if 'loss' not in monitor_key else float('inf')
            for ckpt in all_checkpoints:
                if 'latest' in ckpt:
                    continue
                try:
                    # Extract score for the specified monitor_key
                    score_str = ckpt.split(f'{monitor_key}=')[1].split('.ckpt')[0]
                    score = float(score_str)
                    
                    # Update best score based on whether we're minimizing or maximizing
                    if 'loss' in monitor_key:
                        if score < best_score:
                            best_ckpt = ckpt
                            best_score = score
                    else:
                        if score > best_score:
                            best_ckpt = ckpt
                            best_score = score
                except (IndexError, ValueError):
                    # Skip checkpoints that don't have the monitor_key
                    continue
            
            if best_ckpt is None:
                raise ValueError(f"No checkpoints found with monitor key: {monitor_key}")
            
            return pathlib.Path(self.output_dir).joinpath('checkpoints', best_ckpt)
        else:
            raise NotImplementedError(f"tag {tag} not implemented")
            

    def load_payload(self, payload, exclude_keys=None, include_keys=None, **kwargs):
        if exclude_keys is None:
            exclude_keys = tuple()
        if include_keys is None:
            include_keys = payload['pickles'].keys()
        for key, value in payload['state_dicts'].items():
            if key not in exclude_keys:
                self.__dict__[key].load_state_dict(value, **kwargs)
        for key in include_keys:
            if key in payload['pickles']:
                self.__dict__[key] = dill.loads(payload['pickles'][key])
    
    def load_checkpoint(self, path=None, tag='latest',
            exclude_keys=None, 
            include_keys=None, 
            **kwargs):
        if path is None:
            path = self.get_checkpoint_path(tag=tag)
        else:
            path = pathlib.Path(path)
        payload = torch.load(path.open('rb'), pickle_module=dill, map_location='cpu')
        self.load_payload(payload, 
            exclude_keys=exclude_keys, 
            include_keys=include_keys)
        return payload
    
    @classmethod
    def create_from_checkpoint(cls, path, 
            exclude_keys=None, 
            include_keys=None,
            **kwargs):
        payload = torch.load(open(path, 'rb'), pickle_module=dill)
        instance = cls(payload['cfg'])
        instance.load_payload(
            payload=payload, 
            exclude_keys=exclude_keys,
            include_keys=include_keys,
            **kwargs)
        return instance

    def save_snapshot(self, tag='latest'):
        """
        Quick loading and saving for reserach, saves full state of the workspace.

        However, loading a snapshot assumes the code stays exactly the same.
        Use save_checkpoint for long-term storage.
        """
        path = pathlib.Path(self.output_dir).joinpath('snapshots', f'{tag}.pkl')
        path.parent.mkdir(parents=False, exist_ok=True)
        torch.save(self, path.open('wb'), pickle_module=dill)
        return str(path.absolute())
    
    @classmethod
    def create_from_snapshot(cls, path):
        return torch.load(open(path, 'rb'), pickle_module=dill)
    

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath('config'))
)
def main(cfg):
    workspace = TrainManiFlowRoboTwinTransformerWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
