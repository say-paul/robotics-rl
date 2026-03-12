# save as: read_sensors.py
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
import time

ChannelFactoryInitialize(1, "lo")

def on_state(msg: LowState_):
    imu = msg.imu_state
    print(f"--- t={time.monotonic():.3f} ---")
    print(f"  IMU quat:  [{imu.quaternion[0]:.4f}, {imu.quaternion[1]:.4f}, {imu.quaternion[2]:.4f}, {imu.quaternion[3]:.4f}]")
    print(f"  IMU gyro:  [{imu.gyroscope[0]:.4f}, {imu.gyroscope[1]:.4f}, {imu.gyroscope[2]:.4f}]")
    print(f"  IMU accel: [{imu.accelerometer[0]:.4f}, {imu.accelerometer[1]:.4f}, {imu.accelerometer[2]:.4f}]")
    print(f"  Motor[0] q={msg.motor_state[0].q:.4f}  dq={msg.motor_state[0].dq:.4f}  tau={msg.motor_state[0].tau_est:.4f}")

sub = ChannelSubscriber("rt/lowstate", LowState_)
sub.Init(on_state, 10)

while True:
    time.sleep(1)