"""
DDS bridge between MuJoCo simulation and unitree_sdk2_python.
Adapted from unitree_mujoco/simulate_python/unitree_sdk2py_bridge.py
with the have_imu/have_frame_sensor attribute bug fixed.
"""

import mujoco
import numpy as np

from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelPublisher

from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import WirelessController_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__SportModeState_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__WirelessController_
from unitree_sdk2py.utils.thread import RecurrentThread

import config

if config.ROBOT == "g1":
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_ as LowState_default
else:
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
    from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowState_ as LowState_default

TOPIC_LOWCMD = "rt/lowcmd"
TOPIC_LOWSTATE = "rt/lowstate"
TOPIC_HIGHSTATE = "rt/sportmodestate"
TOPIC_WIRELESS_CONTROLLER = "rt/wirelesscontroller"

MOTOR_SENSOR_NUM = 3
NUM_MOTOR_IDL_GO = 20
NUM_MOTOR_IDL_HG = 35


class UnitreeSdk2Bridge:

    def __init__(self, mj_model, mj_data):
        self.mj_model = mj_model
        self.mj_data = mj_data

        self.num_motor = self.mj_model.nu
        self.dim_motor_sensor = MOTOR_SENSOR_NUM * self.num_motor
        self.have_imu = False
        self.have_frame_sensor = False
        self.dt = self.mj_model.opt.timestep
        self.idl_type = self.num_motor > NUM_MOTOR_IDL_GO

        for i in range(self.dim_motor_sensor, self.mj_model.nsensor):
            name = mujoco.mj_id2name(
                self.mj_model, mujoco._enums.mjtObj.mjOBJ_SENSOR, i
            )
            if name == "imu_quat":
                self.have_imu = True
            if name == "frame_pos":
                self.have_frame_sensor = True

        self.low_state = LowState_default()
        self.low_state_puber = ChannelPublisher(TOPIC_LOWSTATE, LowState_)
        self.low_state_puber.Init()
        self.lowStateThread = RecurrentThread(
            interval=self.dt, target=self.PublishLowState, name="sim_lowstate"
        )
        self.lowStateThread.Start()

        self.high_state = unitree_go_msg_dds__SportModeState_()
        self.high_state_puber = ChannelPublisher(TOPIC_HIGHSTATE, SportModeState_)
        self.high_state_puber.Init()
        self.HighStateThread = RecurrentThread(
            interval=self.dt, target=self.PublishHighState, name="sim_highstate"
        )
        self.HighStateThread.Start()

        self.wireless_controller = unitree_go_msg_dds__WirelessController_()
        self.wireless_controller_puber = ChannelPublisher(
            TOPIC_WIRELESS_CONTROLLER, WirelessController_
        )
        self.wireless_controller_puber.Init()
        self.WirelessControllerThread = RecurrentThread(
            interval=0.01,
            target=self.PublishWirelessController,
            name="sim_wireless_controller",
        )
        self.WirelessControllerThread.Start()

        self.low_cmd_suber = ChannelSubscriber(TOPIC_LOWCMD, LowCmd_)
        self.low_cmd_suber.Init(self.LowCmdHandler, 10)

    def LowCmdHandler(self, msg: LowCmd_):
        if self.mj_data is None:
            return
        for i in range(self.num_motor):
            self.mj_data.ctrl[i] = (
                msg.motor_cmd[i].tau
                + msg.motor_cmd[i].kp
                * (msg.motor_cmd[i].q - self.mj_data.sensordata[i])
                + msg.motor_cmd[i].kd
                * (msg.motor_cmd[i].dq - self.mj_data.sensordata[i + self.num_motor])
            )

    def PublishLowState(self):
        if self.mj_data is None:
            return
        for i in range(self.num_motor):
            self.low_state.motor_state[i].q = self.mj_data.sensordata[i]
            self.low_state.motor_state[i].dq = self.mj_data.sensordata[
                i + self.num_motor
            ]
            self.low_state.motor_state[i].tau_est = self.mj_data.sensordata[
                i + 2 * self.num_motor
            ]

        if self.have_frame_sensor:
            s = self.dim_motor_sensor
            self.low_state.imu_state.quaternion[0] = self.mj_data.sensordata[s + 0]
            self.low_state.imu_state.quaternion[1] = self.mj_data.sensordata[s + 1]
            self.low_state.imu_state.quaternion[2] = self.mj_data.sensordata[s + 2]
            self.low_state.imu_state.quaternion[3] = self.mj_data.sensordata[s + 3]

            self.low_state.imu_state.gyroscope[0] = self.mj_data.sensordata[s + 4]
            self.low_state.imu_state.gyroscope[1] = self.mj_data.sensordata[s + 5]
            self.low_state.imu_state.gyroscope[2] = self.mj_data.sensordata[s + 6]

            self.low_state.imu_state.accelerometer[0] = self.mj_data.sensordata[s + 7]
            self.low_state.imu_state.accelerometer[1] = self.mj_data.sensordata[s + 8]
            self.low_state.imu_state.accelerometer[2] = self.mj_data.sensordata[s + 9]

        self.low_state_puber.Write(self.low_state)

    def PublishHighState(self):
        if self.mj_data is None:
            return
        s = self.dim_motor_sensor
        self.high_state.position[0] = self.mj_data.sensordata[s + 10]
        self.high_state.position[1] = self.mj_data.sensordata[s + 11]
        self.high_state.position[2] = self.mj_data.sensordata[s + 12]
        self.high_state.velocity[0] = self.mj_data.sensordata[s + 13]
        self.high_state.velocity[1] = self.mj_data.sensordata[s + 14]
        self.high_state.velocity[2] = self.mj_data.sensordata[s + 15]
        self.high_state_puber.Write(self.high_state)

    def PublishWirelessController(self):
        pass

    def PrintSceneInformation(self):
        print("\n<<--- Links --->>")
        for i in range(self.mj_model.nbody):
            name = mujoco.mj_id2name(self.mj_model, mujoco._enums.mjtObj.mjOBJ_BODY, i)
            if name:
                print(f"  [{i}] {name}")

        print("\n<<--- Joints --->>")
        for i in range(self.mj_model.njnt):
            name = mujoco.mj_id2name(self.mj_model, mujoco._enums.mjtObj.mjOBJ_JOINT, i)
            if name:
                print(f"  [{i}] {name}")

        print("\n<<--- Actuators --->>")
        for i in range(self.mj_model.nu):
            name = mujoco.mj_id2name(self.mj_model, mujoco._enums.mjtObj.mjOBJ_ACTUATOR, i)
            if name:
                print(f"  [{i}] {name}")

        print(f"\n  Motors: {self.num_motor}  |  Sensors: {self.mj_model.nsensor}"
              f"  |  IMU: {self.have_imu}  |  Frame: {self.have_frame_sensor}\n")
