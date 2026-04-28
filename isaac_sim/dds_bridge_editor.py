# This project was developed with assistance from AI tools.
#
# Isaac Sim Script Editor version of the DDS bridge.
# Paste this into the Script Editor window inside a running Isaac Sim.
#
# Prerequisites:
#   1. Load a G1 USD scene (or use create_g1_scene.py output)
#   2. pip install cyclonedds in Isaac Sim's Python env
#   3. Update SCRIPTS_DIR below to point to your robotics-rl/scripts dir

import sys
import time
import threading

import numpy as np

# --- Configuration ---
SCRIPTS_DIR = "/home/jary/redhat/git/robotics-rl/scripts"
DOMAIN_ID = 0
STATE_TOPIC = "rt/g1/state"
COMMAND_TOPIC = "rt/g1/command"
ROBOT_PRIM = "/World/G1"

# --- Setup imports ---
sys.path.insert(0, SCRIPTS_DIR)

from omni.isaac.dynamic_control import _dynamic_control
from pxr import UsdPhysics
import omni.usd
import omni.timeline

from cyclonedds.domain import DomainParticipant
from cyclonedds.sub import DataReader
from cyclonedds.pub import DataWriter
from cyclonedds.topic import Topic
from engines.dds_types import RobotStateDDS, JointCommandDDS

# --- Find articulation ---
stage = omni.usd.get_context().get_stage()
articulation_prims = []
for prim in stage.Traverse():
    if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        articulation_prims.append(str(prim.GetPath()))

robot_prim_path = ROBOT_PRIM
if articulation_prims:
    print(f"Articulations found: {articulation_prims}")
    if robot_prim_path not in articulation_prims:
        robot_prim_path = articulation_prims[0]
        print(f"  Using: {robot_prim_path}")
else:
    print("No articulations found in stage. Load a robot USD first.")
    raise SystemExit

# --- Start physics ---
timeline = omni.timeline.get_timeline_interface()
if not timeline.is_playing():
    timeline.play()

dc = _dynamic_control.acquire_dynamic_control_interface()
import omni.kit.app
for _ in range(10):
    omni.kit.app.get_app().update()

art_handle = dc.get_articulation(robot_prim_path)
n_dofs = dc.get_articulation_dof_count(art_handle)
print(f"Robot: {robot_prim_path}, {n_dofs} DOFs")

# --- DDS ---
dp = DomainParticipant(domain_id=DOMAIN_ID)
state_topic = Topic(dp, STATE_TOPIC, RobotStateDDS)
command_topic = Topic(dp, COMMAND_TOPIC, JointCommandDDS)
state_writer = DataWriter(dp, state_topic)
command_reader = DataReader(dp, command_topic)
print(f"DDS domain={DOMAIN_ID}, state={STATE_TOPIC}, cmd={COMMAND_TOPIC}")

# --- Bridge loop (runs in background thread) ---
_running = True

def bridge_loop():
    step = 0
    while _running:
        root_body = dc.get_articulation_root_body(art_handle)
        if root_body == _dynamic_control.INVALID_HANDLE:
            time.sleep(0.01)
            continue

        pose = dc.get_rigid_body_pose(root_body)
        root_pos = np.array([pose.p.x, pose.p.y, pose.p.z])
        root_quat_wxyz = np.array([pose.r.w, pose.r.x, pose.r.y, pose.r.z])

        lin_vel = dc.get_rigid_body_linear_velocity(root_body)
        ang_vel = dc.get_rigid_body_angular_velocity(root_body)
        root_lin = np.array([lin_vel.x, lin_vel.y, lin_vel.z])
        root_ang = np.array([ang_vel.x, ang_vel.y, ang_vel.z])

        dof_states = dc.get_articulation_dof_states(art_handle, _dynamic_control.STATE_ALL)
        joint_pos = dof_states["pos"]
        joint_vel = dof_states["vel"]

        qpos = np.concatenate([root_pos, root_quat_wxyz, joint_pos])
        qvel = np.concatenate([root_lin, root_ang, joint_vel])

        state_writer.write(RobotStateDDS(
            qpos=qpos.tolist(), qvel=qvel.tolist(),
            timestamp_ns=int(time.time() * 1e9)))

        samples = command_reader.take(N=100)
        if samples:
            cmd = samples[-1]
            targets = np.array(cmd.target_positions, dtype=np.float32)
            for i in range(min(len(targets), n_dofs)):
                dc.set_dof_position_target(
                    dc.get_articulation_dof(art_handle, i), float(targets[i]))

        step += 1
        if step % 500 == 0:
            print(f"[bridge] step={step} z={root_pos[2]:.3f}")

        time.sleep(0.002)

_thread = threading.Thread(target=bridge_loop, daemon=True)
_thread.start()
print("DDS bridge running in background. To stop: _running = False")
