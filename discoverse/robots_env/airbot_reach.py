import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from discoverse.envs import SimulatorBase
from discoverse.utils.base_config import BaseConfig
import argparse
import torch

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

class AirbotPlayBase(SimulatorBase):
    def __init__(self, config: AirbotPlayCfg):
        self.nj = 7
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
        print("    arm .ctrl  = {}".format(np.array2string(self.mj_data.ctrl[:self.nj], separator=', ')))
        print("    arm .force = {}".format(np.array2string(self.sensor_joint_force, separator=', ')))

        print("    sensor end posi  = {}".format(np.array2string(self.sensor_endpoint_posi_local, separator=', ')))
        print("    sensor end euler = {}".format(np.array2string(Rotation.from_quat(self.sensor_endpoint_quat_local[[1,2,3,0]]).as_euler("xyz"), separator=', ')))

    def resetState(self):
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        self.mj_data.qpos[:self.nj] = self.init_joint_pose.copy()
        self.mj_data.ctrl[:self.nj] = self.init_joint_ctrl.copy()
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def updateControl(self, action):
        if self.mj_data.qpos[self.nj-1] < 0.0:
            self.mj_data.qpos[self.nj-1] = 0.0
        self.mj_data.ctrl[:self.nj] = np.clip(action[:self.nj], self.mj_model.actuator_ctrlrange[:self.nj,0], self.mj_model.actuator_ctrlrange[:self.nj,1])

    def checkTerminated(self):
        return False

    def getObservation(self):
        self.obs = {
            # "time" : self.mj_data.time,
            "jq"   : self.sensor_joint_qpos.tolist()[:6],
            "jv"   : self.sensor_joint_qvel.tolist()[:6],
            # "jf"   : self.sensor_joint_force.tolist(),
            "ep"   : self.sensor_endpoint_posi_local.tolist(),
            "eq"   : self.sensor_endpoint_quat_local.tolist(),
            "action":np.zeros(6),
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

    action = np.array([-0.79708582 ,-0.29467133  ,0.14392811 , 0.27541178 , 0.12742431 , 0.9054758,0])
    prev_action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    policy = torch.jit.load(args.load_model)

    # 执行控制命令
    while exec_node.running:
        # 将关节角命令传递给step方法
        obs, pri_obs, rew, ter, info = exec_node.step(action)
        # obs["action"]=prev_action[:6]

        # obs_data = np.concatenate([
        #     np.array(obs["jq"]),  # 关节位置
        #     np.array(obs["jv"]),  # 关节速度
        #     np.array(obs["ep"]),  # 末端执行器位置
        #     np.array(obs["eq"]),   # 末端执行器四元数
        #     np.array(obs["action"])  # 上一时刻的动作
        # ])

        # obs_tensor = torch.tensor(obs_data, dtype=torch.float32)
        # obs_flattened = torch.flatten(obs_tensor, start_dim=0)

        # action[:6] = policy(obs_flattened)[0].detach().numpy()

        # prev_action = action.copy()
        print("action", action[:6])
        # 如果需要打印机器人状态
        # exec_node.printMessage()
