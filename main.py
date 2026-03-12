"""
Unitree G1 MuJoCo Simulation with browser streaming and harness control.

Usage:
    MUJOCO_GL=egl python3 main.py [--port 8888] [--width 1280] [--height 720]
"""

import os
os.environ["MUJOCO_GL"] = "egl"

import argparse
from threading import Thread

import mujoco

import config
from sim_context import SimContext
from renderer import FrameBuffer, run_renderer
from sim_engine import run_simulation
from web import create_app


def main():
    parser = argparse.ArgumentParser(description="G1 Harness Training Simulation")
    parser.add_argument("--port", type=int, default=8888)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    print(f"[init] Loading model: {config.ROBOT_SCENE}")
    mj_model = mujoco.MjModel.from_xml_path(config.ROBOT_SCENE)
    mj_data = mujoco.MjData(mj_model)
    mj_model.opt.timestep = config.SIMULATE_DT

    ctx = SimContext(mj_model, mj_data)
    frames = FrameBuffer()

    Thread(target=run_simulation, args=(ctx,), daemon=True).start()
    Thread(target=run_renderer, args=(ctx, frames, args.width, args.height), daemon=True).start()

    app = create_app(ctx, frames)

    print(f"\n{'='*60}")
    print(f"  G1 Harness Training Simulation")
    print(f"  Browser: http://0.0.0.0:{args.port}")
    print(f"{'='*60}\n")

    try:
        app.run(host="0.0.0.0", port=args.port, threaded=True)
    except KeyboardInterrupt:
        pass
    finally:
        ctx.shutdown()
        print("[shutdown] Done.")


if __name__ == "__main__":
    main()
