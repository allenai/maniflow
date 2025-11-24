
from .base_task import Base_task
from .utils import *
import numpy as np
import sapien

class mug_hanging(Base_task):
    def setup_demo(self,is_test=False,**kwags):
        super()._init(**kwags)
        self.create_table_and_wall()
        self.load_robot()
        self.setup_planner()
        self.load_camera(kwags.get('camera_w', 640),kwags.get('camera_h', 480))
        self.pre_move()
        if is_test:
            self.id_list = [1,3,4,6,7,8,9]
        else:
            self.id_list = [0,2]
        self.load_actors()
        self.step_lim = 800
    
    def pre_move(self):
        render_freq = self.render_freq
        self.render_freq=0
        self.together_open_gripper(save_freq=None)
        self.render_freq = render_freq

    def load_actors(self):
        self.mug, self.mug_data = rand_create_glb(
            self.scene,
            xlim=[-0.25,-0.1],
            ylim=[-0.05,0.1],
            zlim=[0.79],
            ylim_prop = True,
            modelname="039_mug",
            rotate_rand=True,
            rotate_lim=[0,1.57,0],
            qpos=[0.707,0.707,0,0],
            convex=False,
            model_id = np.random.choice(self.id_list)
        )

        rack_pose = rand_pose(
            xlim=[0.1,0.3], 
            ylim = [0.05,0.2],
            zlim=[0.745],
            rotate_rand=False,
            qpos=[-0.22, -0.22, 0.67, 0.67]
        )
        self.rack, self.rack_data = create_obj(
            self.scene,
            pose = rack_pose,
            modelname="040_rack",
            is_static=True,
            convex=True
        )
        self.mug.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent).mass = 0.001

    def play_once(self):
        pass

    def check_success(self):
        mug_target_pose = self.get_actor_goal_pose(self.mug,self.mug_data)
        eps = np.array([0.01,0.01,0.01])
        return np.all(abs(mug_target_pose - self.rack.get_pose().p + [0.02,0.02,-0.1]) < eps) and np.all(abs(self.right_endpose.global_pose.p - [0.3,-0.32,0.935]) < 0.05)
