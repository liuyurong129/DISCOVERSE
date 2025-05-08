import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from discoverse.envs import SimulatorBase
from discoverse.utils.base_config import BaseConfig
import argparse
import torch
import random
import math

class AirbotPlayCfg(BaseConfig):
    mjcf_file_path = "mjcf/airbot_play_floor.xml"
    decimation     = 4
    timestep       = 0.005
    sync           = True
    headless       = False
    init_key       = "home"
    render_set     = {
        "fps"    : 30,
        "width"  : 1280,
        "height" : 720,
    }
    obs_rgb_cam_id  = None
    rb_link_list   = ["arm_base", "link1", "link2", "link3", "link4", "link5", "link6", "right", "left"]
    obj_list       = []
    use_gaussian_renderer = False

class ReachTaskConfig:
    """Configuration for the reaching task"""
    def __init__(self):
        # Target position ranges (similar to your command ranges)
        self.pos_range_x = (0.35, 0.65)
        self.pos_range_y = (-0.2, 0.2)
        self.pos_range_z = (0.15, 0.5)
        self.pos_range_roll=(0,0)
        self.pos_range_pitch=(math.pi,math.pi)
        self.pos_range_yaw=(-math.pi/2, math.pi/2)
        
        # Current target position
        self.target_pos = np.array([
            random.uniform(*self.pos_range_x),
            random.uniform(*self.pos_range_y),
            random.uniform(*self.pos_range_z)
        ])
        self.target_rpy = np.array([
            random.uniform(*self.pos_range_roll),
            random.uniform(*self.pos_range_pitch),
            random.uniform(*self.pos_range_yaw)
        ])

        # Target update frequency (in seconds)
        self.target_update_time = 4.0
        self.time_since_last_update = 0.0
    
    def update_target(self, dt):
        """Update target position if needed"""
        self.time_since_last_update += dt
        if self.time_since_last_update >= self.target_update_time:
            self.target_pos[0] = random.uniform(*self.pos_range_x)
            self.target_pos[1] = random.uniform(*self.pos_range_y)
            self.target_pos[2] = random.uniform(*self.pos_range_z)

            self.target_rpy[0] = random.uniform(*self.pos_range_roll)
            self.target_rpy[1] = random.uniform(*self.pos_range_pitch)
            self.target_rpy[2] = random.uniform(*self.pos_range_yaw)
            self.time_since_last_update = 0.0
            return True
        return False

class AirbotPlayBase(SimulatorBase):
    def __init__(self, config: AirbotPlayCfg):
        self.nj = 7
        self.reach_task = ReachTaskConfig()  # Initialize the reach task config
        super().__init__(config)

    def post_load_mjcf(self):
        try:
            self.init_joint_pose = self.mj_model.key(self.config.init_key).qpos[:self.nj]
            self.init_joint_ctrl = self.mj_model.key(self.config.init_key).ctrl[:self.nj]
        except KeyError as e:
            self.init_joint_pose = np.zeros(self.nj)
            self.init_joint_ctrl = np.zeros(self.nj)

        self.sensor_joint_qpos = self.mj_data.sensordata[:self.nj]
        self.sensor_joint_qvel = self.mj_data.sensordata[self.nj:2*self.nj]
        self.sensor_joint_force = self.mj_data.sensordata[2*self.nj:3*self.nj]
        self.sensor_endpoint_posi_local = self.mj_data.sensordata[3*self.nj:3*self.nj+3]
        self.sensor_endpoint_quat_local = self.mj_data.sensordata[3*self.nj+3:3*self.nj+7]
        self.sensor_endpoint_linear_vel_local = self.mj_data.sensordata[3*self.nj+7:3*self.nj+10]
        self.sensor_endpoint_gyro = self.mj_data.sensordata[3*self.nj+10:3*self.nj+13]
        self.sensor_endpoint_acc = self.mj_data.sensordata[3*self.nj+13:3*self.nj+16]

    def printMessage(self):
        print("-" * 100)
        print("mj_data.time  = {:.3f}".format(self.mj_data.time))
        print("    arm .qpos  = {}".format(np.array2string(self.sensor_joint_qpos, separator=', ')))
        print("    arm .qvel  = {}".format(np.array2string(self.sensor_joint_qvel, separator=', ')))
        # print("    arm .force = {}".format(np.array2string(self.sensor_joint_force, separator=', ')))

        # print("    sensor end posi  = {}".format(np.array2string(self.sensor_endpoint_posi_local, separator=', ')))
        # print("    sensor end euler = {}".format(np.array2string(Rotation.from_quat(self.sensor_endpoint_quat_local[[1,2,3,0]]).as_euler("xyz"), separator=', ')))
        
        print("    target position  = {}".format(np.array2string(self.reach_task.target_pos, separator=', ')))
        print("    target orientation = {}".format(np.array2string(self.reach_task.target_rpy, separator=', ')))
        print("    arm .ctrl  = {}".format(np.array2string(self.mj_data.ctrl[:self.nj], separator=', ')))

    def resetState(self):
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        self.mj_data.qpos[:self.nj] = self.init_joint_pose.copy()
        self.mj_data.ctrl[:self.nj] = self.init_joint_ctrl.copy()
        # Reset the reach task with a new random target
        self.reach_task = ReachTaskConfig()
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def updateControl(self, action):
        if self.mj_data.qpos[self.nj-1] < 0.0:
            self.mj_data.qpos[self.nj-1] = 0.0
        self.mj_data.ctrl[:self.nj] = np.clip(action[:self.nj], self.mj_model.actuator_ctrlrange[:self.nj,0], self.mj_model.actuator_ctrlrange[:self.nj,1])

    def checkTerminated(self):
        return False

    def getObservation(self):
        # Update target position based on time
        target_updated = self.reach_task.update_target(self.config.timestep)
        
        # Convert target RPY to quaternion for observation
        target_quat = Rotation.from_euler('xyz', self.reach_task.target_rpy).as_quat()
        # Rearrange to match the format used in the original code [w,x,y,z]
        target_quat_wxyz = np.array([target_quat[3], target_quat[0], target_quat[1], target_quat[2]])
        
        self.obs = {
            # "time" : self.mj_data.time,
            "jq"   : self.sensor_joint_qpos.tolist()[:6],
            "jv"   : self.sensor_joint_qvel.tolist()[:6],
            # "jf"   : self.sensor_joint_force.tolist(),
            "ep"   : self.reach_task.target_pos.tolist(),  # Use target position instead of actual end-effector
            "eq"   : target_quat_wxyz.tolist(),  # Use target orientation instead of actual end-effector
            "action": np.zeros(6),
            # "img"  : self.img_rgb_obs_s.copy(),
            # "depth" : self.img_depth_obs_s.copy()
        }
        return self.obs

    def getPrivilegedObservation(self):
        return self.obs

    def getReward(self):
        return None

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='AirBot Reach Task Deployment')
    parser.add_argument('--load_model', type=str, required=True,
                      help='Path to the trained policy model')

    args = parser.parse_args()

    cfg = AirbotPlayCfg()
    exec_node = AirbotPlayBase(cfg)

    obs = exec_node.reset()

    action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    prev_action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    policy = torch.jit.load(args.load_model)

    # 执行控制命令
    while exec_node.running:
        # 将关节角命令传递给step方法
        obs, pri_obs, rew, ter, info = exec_node.step(action)
        obs["action"] = prev_action[:6]

        obs_data = np.concatenate([
            np.array(obs["jq"]),  # 关节位置
            np.array(obs["jv"]),  # 关节速度
            np.array(obs["ep"]),  # 目标位置（不是末端执行器实际位置）
            np.array(obs["eq"]),  # 目标朝向（不是末端执行器实际朝向）
            np.array(obs["action"])  # 上一时刻的动作
        ])

        obs_tensor = torch.tensor(obs_data, dtype=torch.float32)
        obs_flattened = torch.flatten(obs_tensor, start_dim=0)

        action[:6] = policy(obs_flattened)[0].detach().numpy()*0.5

        prev_action = action.copy()
        print("action", action)
        # 如果需要打印机器人状态
        exec_node.printMessage()