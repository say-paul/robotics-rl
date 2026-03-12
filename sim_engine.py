"""
MuJoCo physics loop and DDS bridge initialisation.

Runs in its own thread, stepping the simulation at the model timestep,
applying harness forces, and publishing/subscribing DDS topics via the bridge.
"""

import time

import mujoco

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from dds_bridge import UnitreeSdk2Bridge

import config


def run_simulation(ctx):
    """
    Physics thread entry point.

    Initialises DDS, creates the bridge, then loops at the model timestep.
    """
    ChannelFactoryInitialize(config.DOMAIN_ID, config.INTERFACE)
    ctx.dds_ready.set()

    bridge = UnitreeSdk2Bridge(ctx.mj_model, ctx.mj_data)

    if config.PRINT_SCENE_INFORMATION:
        bridge.PrintSceneInformation()

    torso_id = ctx.mj_model.body("torso_link").id

    print(
        f"[sim] Physics running  dt={ctx.mj_model.opt.timestep}s"
        f"  harness -> body '{ctx.mj_model.body(torso_id).name}'"
    )

    while ctx.sim_running:
        step_start = time.perf_counter()

        with ctx.physics_lock:
            pos = ctx.mj_data.qpos[:3].copy()
            vel = ctx.mj_data.qvel[:3].copy()
            ctx.robot_pos[:] = pos

            force = ctx.harness.compute_force(pos, vel)
            ctx.mj_data.xfrc_applied[torso_id, :3] = force

            mujoco.mj_step(ctx.mj_model, ctx.mj_data)

        dt = ctx.mj_model.opt.timestep - (time.perf_counter() - step_start)
        if dt > 0:
            time.sleep(dt)
