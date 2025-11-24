
from .base_task import Base_task
from .utils import *
import sapien

class dual_bottles_pick_easy(Base_task):
    def setup_demo(self,**kwags):
        super()._init(**kwags)
        self.create_table_and_wall()
        self.load_robot()
        self.setup_planner()
        self.load_camera(kwags.get('camera_w', 640),kwags.get('camera_h', 480))
        self.pre_move()
        self.load_actors()
        self.step_lim = 400
    
    def pre_move(self):
        render_freq = self.render_freq
        self.render_freq=0
        self.together_open_gripper(save_freq=None)
        self.render_freq = render_freq

    def load_actors(self, **kwargs):
        self.red_bottle,_ = rand_create_glb(
            self.scene,
            xlim=[-0.25,-0.05],
            ylim=[0.03,0.23],
            zlim=[0.865],
            modelname="001_bottles",
            rotate_rand=False,
            qpos=[0.707,0.707,0,0],
            scale=(0.132,0.132,0.132),
            model_id=13
        )

        self.green_bottle, _=rand_create_glb(
            self.scene,
            xlim=[0.05,0.25],
            ylim=[0.03,0.23],
            zlim=[0.865],
            modelname="001_bottles",
            rotate_rand=False,
            qpos=[0.707,0.707,0,0],
            scale=(0.161,0.161,0.161),
            model_id=16
        )

        self.red_bottle.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent).mass = 0.01
        self.green_bottle.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent).mass = 0.01

        render_freq = self.render_freq
        self.render_freq = 0
        for _ in range(4):
            self.together_open_gripper(save_freq=None)
        self.render_freq = render_freq

    def play_once(self):
        pass

    def check_success(self):
        red_target = [-0.046,-0.105]
        green_target = [0.057,-0.105]
        eps = 0.03
        red_bottle_pose = self.red_bottle.get_pose().p
        green_bottle_pose = self.green_bottle.get_pose().p
        if red_bottle_pose[2] < 0.78 or green_bottle_pose[2] < 0.78:
            self.actor_pose = False
        return abs(red_bottle_pose[0]-red_target[0])<eps and abs(red_bottle_pose[1]-red_target[1])<eps and red_bottle_pose[2]>0.9 and \
               abs(green_bottle_pose[0]-green_target[0])<eps and abs(green_bottle_pose[1]-green_target[1])<eps and green_bottle_pose[2]>0.9
