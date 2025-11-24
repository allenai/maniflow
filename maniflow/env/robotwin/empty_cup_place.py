
from .base_task import Base_task
from .utils import *
import sapien

class empty_cup_place(Base_task):
    def setup_demo(self,**kwags):
        super()._init(**kwags)
        self.create_table_and_wall()
        self.load_robot()
        self.setup_planner()
        self.load_camera(kwags.get('camera_w', 640),kwags.get('camera_h', 480))
        self.pre_move()
        self.load_actors()
        self.step_lim = 500
    
    def pre_move(self):
        render_freq = self.render_freq
        self.render_freq=0
        self.together_open_gripper(save_freq=None)
        self.render_freq = render_freq

    def load_actors(self):
        tag = np.random.randint(0,2)
        if tag==0:
            self.cup,self.cup_data = rand_create_glb(
                self.scene,
                xlim=[0.15,0.3],
                ylim=[-0.2,0.05],
                zlim=[0.8],
                modelname="022_cup",
                rotate_rand=False,
                qpos=[0.707,0.707,0,0],
            )
            cup_pose = self.cup.get_pose().p

            coaster_pose = rand_pose(
                xlim=[-0.05,0.1],
                ylim=[-0.2,0.05],
                zlim=[0.76],
                rotate_rand=False,
                qpos=[0.707,0.707,0,0],
            )

            while np.sum(pow(cup_pose[:2] - coaster_pose.p[:2],2)) < 0.01:
                coaster_pose = rand_pose(
                    xlim=[-0.05,0.1],
                    ylim=[-0.2,0.05],
                    zlim=[0.76],
                    rotate_rand=False,
                    qpos=[0.707,0.707,0,0],
                )
            self.coaster,_ = create_obj(
                self.scene,
                pose = coaster_pose,
                modelname="019_coaster",
                convex=True
            )
        else:
            self.cup,self.cup_data = rand_create_glb(
                self.scene,
                xlim=[-0.3,-0.15],
                ylim=[-0.2,0.05],
                zlim=[0.8],
                modelname="022_cup",
                rotate_rand=False,
                qpos=[0.707,0.707,0,0],
            )
            cup_pose = self.cup.get_pose().p

            coaster_pose = rand_pose(
                xlim=[-0.1, 0.05],
                ylim=[-0.2,0.05],
                zlim=[0.76],
                rotate_rand=False,
                qpos=[0.707,0.707,0,0],
            )

            while np.sum(pow(cup_pose[:2] - coaster_pose.p[:2],2)) < 0.01:
                coaster_pose = rand_pose(
                    xlim=[-0.1, 0.05],
                    ylim=[-0.2,0.05],
                    zlim=[0.76],
                    rotate_rand=False,
                    qpos=[0.707,0.707,0,0],
                )
            self.coaster,_ = create_obj(
                self.scene,
                pose = coaster_pose,
                modelname="019_coaster",
                convex=True
            )

        
        self.cup.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent).mass = 0.01
        self.coaster.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent).mass = 0.01

    def play_once(self):
        pass
    
    def check_success(self):
        eps = 0.025
        coaster_pose = self.coaster.get_pose().p
        cup_pose = self.cup.get_pose().p
        return abs(cup_pose[0] - coaster_pose[0])<eps  and  abs(cup_pose[1] - coaster_pose[1])<eps and (cup_pose[2] - 0.792) < 0.005
