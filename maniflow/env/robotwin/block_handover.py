
from .base_task import Base_task
from .utils import *
import sapien
import math

class block_handover(Base_task):
    def setup_demo(self,**kwags):
        super()._init(**kwags)
        self.create_table_and_wall()
        self.load_robot()
        self.setup_planner()
        self.load_camera(kwags.get('camera_w', 640),kwags.get('camera_h', 480))
        self.pre_move()
        self.load_actors()
        self.step_lim = 600
    
    def pre_move(self):
        render_freq = self.render_freq
        self.render_freq=0
        self.together_open_gripper(save_freq=None)

        self.render_freq = render_freq

    def load_actors(self):
        rand_pos = rand_pose(
            xlim=[-0.25,-0.05],
            ylim=[0.,0.25],
            zlim=[0.842],
            qpos=[-0.906,0,0,-0.424]
        )
        self.box = create_box(
            scene = self.scene,
            pose = rand_pos,
            half_size=(0.03,0.03,0.1),
            color=(1,0,0),
            name="box"
        )

        rand_pos = rand_pose(
            xlim=[0.23,0.23],
            ylim=[0.09,0.09],
            zlim=[0.74],
        )

        self.target = create_box(
            scene = self.scene,
            pose = rand_pos,
            half_size=(0.05,0.05,0.005),
            color=(0,0,1),
            name="box"
        )
        self.target.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent).mass = 1
        self.box.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent).mass = 0.1

    def play_once(self):
        pass

    def check_success(self):
        box_pos = self.box.get_pose().p
        target_pose = self.target.get_pose().p
        if box_pos[2] < 0.78:
            self.actor_pose = False
        eps = 0.02
        right_endpose = self.get_right_endpose_pose()
        endpose_target_pose = [0.241,-0.129,0.889,0,-0.7,-0.71,0]
        return abs(box_pos[0] - target_pose[0]) < eps and abs(box_pos[1] - target_pose[1]) < eps and abs(box_pos[2] - 0.85) < 0.0015 and\
               np.all(abs(np.array(right_endpose.p.tolist() + right_endpose.q.tolist()) - endpose_target_pose ) < 0.2 * np.ones(7)) and self.is_right_gripper_open()