from typing import List
import numpy as np
import tqdm
from maniflow.env_runner.base_runner import BaseRunner
from maniflow.env_runner.metaworld_runner import MetaworldRunner
from maniflow.policy.base_policy import BasePolicy

class MetaworldMultitaskRunner(BaseRunner):
    def __init__(self,
                output_dir,
                task_names: List[str],
                eval_episodes=20,
                max_steps=1000,
                n_obs_steps=8,
                n_action_steps=8,
                fps=10,
                crf=22,
                render_size=84,
                tqdm_interval_sec=5.0,
                n_envs=None,
                n_train=None,
                n_test=None,
                device="cuda:0",
                use_point_crop=True,
                num_points=512,
                **kwargs
                ):
        super().__init__(output_dir)
        
        self.task_names = task_names
        self.runners = {}
        
        # Create individual runners for each task
        for task_name in task_names:
            self.runners[task_name] = MetaworldRunner(
                output_dir=output_dir,
                eval_episodes=eval_episodes,
                max_steps=max_steps,
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                fps=fps,
                crf=crf,
                render_size=render_size,
                n_envs=n_envs,
                n_train=n_train,
                n_test=n_test,
                tqdm_interval_sec=tqdm_interval_sec,
                task_name=task_name,
                device=device,
                use_point_crop=use_point_crop,
                num_points=num_points,
            )
            
        self.eval_episodes = eval_episodes
        self.tqdm_interval_sec = tqdm_interval_sec
        
    def run(self, policy: BasePolicy, save_video=True):
        results = {}
        avg_success_rate = 0.0
        
        for task_name in tqdm.tqdm(self.task_names, 
                                 desc="Evaluating tasks",
                                 leave=False,
                                 mininterval=self.tqdm_interval_sec):
            # Run evaluation for this task
            runner = self.runners[task_name]
            task_results = runner.run(policy, save_video)
            results[task_name] = task_results
            
            # Calculate success rate for this task
            success_rate = np.mean(task_results.get('mean_success_rates', 0.0))
            avg_success_rate += success_rate
            
        # Calculate average success rate across all tasks
        avg_success_rate /= len(self.task_names)
        
        results['average_success_rate'] = avg_success_rate

        return results 