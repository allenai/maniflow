
from .base_task import Base_task
from .utils import *
import math
import sapien
import os


class pick_apple_messy(Base_task):
    def setup_demo(self,is_test = False, **kwags):
        super()._init(**kwags)
        self.create_table_and_wall()
        self.load_robot()
        self.setup_planner()
        self.load_camera(kwags.get('camera_w', 640),kwags.get('camera_h', 480))
        self.pre_move()
        if is_test:
            self.id_list = [2*i+1 for i in range(5)]
        else:
            self.id_list = [2*i for i in range(5)]
        self.load_actors()
        self.step_lim = 250
    
    def pre_move(self):
        render_freq = self.render_freq
        self.render_freq=0
        self.together_open_gripper(save_freq=None)
        self.render_freq = render_freq

    def load_actors(self):
        self.actor_list=[]
        self.actor_data_list=[]
        # file_path = 'maniflow/env/robotwin/utils/rand_model_data.json'
        # Get the directory where the current script (pick_apple_messy.py) is located
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Create path to the JSON file relative to the current script
        file_path = os.path.join(current_dir, 'utils', 'rand_model_data.json')
        
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        pose_list = []
        obj_num = 4
        obj_list = np.random.choice([i for i in range(data["number"])], size = obj_num, replace=True)

        apple_pose = rand_pose(
            xlim=[-0.25,0.25],
            ylim=[-0.1,0.1],
            zlim=[0.78],
            ylim_prop=True,
            rotate_rand=False,
        )
        pose_list.append(apple_pose)
        self.apple, self.apple_data = create_obj(
            self.scene,
            pose=apple_pose, # the z value (0.783) is important
            modelname="035_apple",
            convex=True
        )
        for i in obj_list:
            model_index = f"model{i}"
            actor_pose = rand_pose(
                xlim=data[model_index]["xlim"],
                ylim=data[model_index]["ylim"],
                zlim=data[model_index]["zlim"],
                ylim_prop=True,
                rotate_rand=True,
                rotate_lim=data[model_index]["rotate_lim"],
                qpos=data[model_index]["init_qpos"]
            )

            tag = True
            while tag:
                tag = False
                for pose in pose_list:
                    if np.sum(pow(pose.p[:2] - actor_pose.p[:2],2)) < 0.0225:
                        tag = True
                        break
                if tag:
                    actor_pose = rand_pose(
                        xlim=data[model_index]["xlim"],
                        ylim=data[model_index]["ylim"],
                        zlim=data[model_index]["zlim"],
                        ylim_prop=True,
                        rotate_rand=True,
                        rotate_lim=data[model_index]["rotate_lim"],
                        qpos=data[model_index]["init_qpos"]
                    )

            pose_list.append(actor_pose)
            model, model_data = create_actor(
                self.scene,
                pose=actor_pose,
                modelname=data[model_index]["name"],
                convex=True,
                model_z_val = data[model_index]["model_z_val"],
            )

            self.actor_list.append(model)
            self.actor_data_list.append(model_data)

    def play_once(self):
        pass

    def check_success(self):
        apple_pose = self.apple.get_pose().p
        return apple_pose[2] > 0.81
