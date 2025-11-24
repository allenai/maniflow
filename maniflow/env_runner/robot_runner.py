import wandb
import numpy as np
import torch
import tqdm

from maniflow.policy.base_policy import BasePolicy
from maniflow.common.pytorch_util import dict_apply
from maniflow.env_runner.base_runner import BaseRunner
import maniflow.common.logger_util as logger_util
from queue import deque
import importlib
import pathlib
from termcolor import cprint


class RobotRunner(BaseRunner):
    def __init__(self,
                 output_dir,
                 eval_episodes=20,
                 max_steps=200,
                 n_obs_steps=8,
                 n_action_steps=8,
                 fps=10,
                 crf=22,
                 render_size=84,
                 tqdm_interval_sec=5.0,
                 task_name=None,
                 use_point_crop=True,
                 task_config=None,
                 ):
        super().__init__(output_dir)
        self.task_name = task_name
        cprint(f"RobotRunner for task {self.task_name}", 'green')

        steps_per_render = max(10 // fps, 1)

        self.eval_episodes = eval_episodes
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec

        self.logger_util_test = logger_util.LargestKRecorder(K=3)
        self.logger_util_test10 = logger_util.LargestKRecorder(K=5)
        self.obs = deque(maxlen=n_obs_steps+1)
        def env_fn(task_name):
            full_module_path = f'maniflow.env.robotwin.{task_name}'
            env_module = importlib.import_module(full_module_path)
            # self.task_conifg = env_module.task_config
            try:
                env_class = getattr(env_module, task_name)
                env_instance = env_class()
            except:
                raise SystemExit("No Task")
            return env_instance
        self.task_conifg = task_config
        self.env = env_fn(task_name)


    def stack_last_n_obs(self, all_obs, n_steps):
        assert(len(all_obs) > 0)
        all_obs = list(all_obs)
        if isinstance(all_obs[0], np.ndarray):
            result = np.zeros((n_steps,) + all_obs[-1].shape, 
                dtype=all_obs[-1].dtype)
            start_idx = -min(n_steps, len(all_obs))
            result[start_idx:] = np.array(all_obs[start_idx:])
            if n_steps > len(all_obs):
                # pad
                result[:start_idx] = result[start_idx]
        elif isinstance(all_obs[0], torch.Tensor):
            result = torch.zeros((n_steps,) + all_obs[-1].shape, 
                dtype=all_obs[-1].dtype)
            start_idx = -min(n_steps, len(all_obs))
            result[start_idx:] = torch.stack(all_obs[start_idx:])
            if n_steps > len(all_obs):
                # pad
                result[:start_idx] = result[start_idx]
        # support str
        elif isinstance(all_obs[0], str):
            return all_obs * n_steps
        else:
            raise RuntimeError(f'Unsupported obs type {type(all_obs[0])}')
        return result
    
    def reset_obs(self):
        self.obs.clear()

    def update_obs(self, current_obs):
        self.obs.append(current_obs)

    def get_n_steps_obs(self):
        assert(len(self.obs) > 0), 'no observation is recorded, please update obs first'

        result = dict()
        for key in self.obs[0].keys():
            result[key] = self.stack_last_n_obs(
                [obs[key] for obs in self.obs],
                self.n_obs_steps
            )
        return result
    
    @torch.no_grad()
    def get_action(self, policy: BasePolicy, observaton=None) -> bool: # by tianxing chen
        if observaton == None:
            print('==== Get empty observation ===')
            return False
        device, dtype = policy.device, policy.dtype
        self.obs.append(observaton) # update
        obs = self.get_n_steps_obs()
        # create obs dict
        np_obs_dict = dict(obs)
        # device transfer
        obs_dict = dict_apply(np_obs_dict, lambda x: torch.from_numpy(x).to(device=device))
        # run policy
        with torch.no_grad():
            obs_dict_input = {}  # flush unused keys
            obs_dict_input['point_cloud'] = obs_dict['point_cloud'].unsqueeze(0)
            obs_dict_input['head_cam'] = obs_dict['head_cam'].unsqueeze(0)
            obs_dict_input['agent_pos'] = obs_dict['agent_pos'].unsqueeze(0)
            obs_dict_input['task_name'] = [self.task_name]
            action_dict = policy.predict_action(obs_dict_input)
            
        # device_transfer
        np_action_dict = dict_apply(action_dict, lambda x: x.detach().to('cpu').numpy())
        action = np_action_dict['action'].squeeze(0)
        return action
    
    def get_seed_list(self):
        with open(f'{pathlib.Path(__file__).parent.parent}/config/robotwin_env/eval_seeds/'+self.task_conifg['task_name']+'.txt', 'r') as file:
            seed_list = file.read().split()
            seed_list = [int(i) for i in seed_list]
        return seed_list

    def run(self, policy: BasePolicy, save_video=True):
        device = policy.device
        dtype = policy.dtype

        all_success_rates = []
        env = self.env
        env.suc = 0
        env.test_num = 0
        seed_list = self.get_seed_list() # 100 seeds
        now_id = 0
        log_data = dict()

        log_data['task_name'] = self.task_name
        
        for episode_idx in tqdm.tqdm(range(self.eval_episodes), desc=f"Eval in Robotwin {self.task_name} Pointcloud Env", leave=False, mininterval=self.tqdm_interval_sec):
            now_seed = seed_list[episode_idx]
            env.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **self.task_conifg)
            env.deploy_policy_online(policy, self, device=device)
            
            if save_video:
                videos = env.get_video() # N, H, W, C numpy array
                # transform to N, C, H, W
                videos = np.transpose(videos, (0, 3, 1, 2)).astype(np.uint8)
                assert len(videos) != 0, 'No video is recorded'

                # videos_wandb = wandb.Video(videos, fps=self.fps, format="mp4")
                videos_wandb = wandb.Video(videos, fps=self.fps, format="mp4")
                log_data[f'sim_video_eval_{now_id}'] = videos_wandb
            now_id += 1
            env.close()
            if env.render_freq:
                env.viewer.close()
            self.reset_obs()
            # clear cache
            torch.cuda.empty_cache()
            print(f"{self.task_name} success rate: {env.suc}/{env.test_num}, current seed: {now_seed}\n")
            is_success = env.info['success']
            all_success_rates.append(is_success)
        
        log_data['mean_success_rates'] = np.mean(all_success_rates)
        log_data['test_mean_score'] = np.mean(all_success_rates)
        cprint(f"test_mean_score for task {self.task_name}: {np.mean(all_success_rates)}", 'green')

        self.logger_util_test.record(np.mean(all_success_rates))
        self.logger_util_test10.record(np.mean(all_success_rates))
        log_data['SR_test_L3'] = self.logger_util_test.average_of_largest_K()
        log_data['SR_test_L5'] = self.logger_util_test10.average_of_largest_K()


        videos = None
        del env
        
        return log_data

        

if __name__ == '__main__':
    test = RobotRunner('./')
    print('ready')