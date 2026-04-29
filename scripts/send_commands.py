#!/usr/bin/env python3
# This project was developed with assistance from AI tools.
"""Keyboard velocity command sender using Unitree SDK DDS.

Publishes to rt/run_command/cmd on DDS channel 1 (matching the
unitree_sim_isaaclab container).

Usage:
    python scripts/send_commands.py
"""

import sys
import tty
import termios
import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_


def main():
    print("Initializing DDS channel 1...")
    result = ChannelFactoryInitialize(1)
    print(f"ChannelFactoryInitialize done")

    pub = ChannelPublisher("rt/run_command/cmd", String_)
    pub.Init()
    print("Publisher initialized on rt/run_command/cmd")
    time.sleep(0.5)

    vx, vy, vyaw, height = 0.0, 0.0, 0.0, 0.8

    print("G1 Velocity Commander (Unitree SDK DDS channel 1)")
    print("=" * 50)
    print("  w/s  — forward/back")
    print("  a/d  — turn left/right")
    print("  q/e  — strafe left/right")
    print("  r/f  — height up/down")
    print("  x    — stop all")
    print("  Ctrl-C — quit")
    print("=" * 50)

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while True:
            ch = sys.stdin.read(1)
            if ch == 'w':
                vx = min(vx + 0.1, 1.0)
            elif ch == 's':
                vx = max(vx - 0.1, -0.6)
            elif ch == 'a':
                vyaw = min(vyaw + 0.2, 1.57)
            elif ch == 'd':
                vyaw = max(vyaw - 0.2, -1.57)
            elif ch == 'q':
                vy = min(vy + 0.1, 0.5)
            elif ch == 'e':
                vy = max(vy - 0.1, -0.5)
            elif ch == 'r':
                height = min(height + 0.05, 0.8)
            elif ch == 'f':
                height = max(height - 0.05, 0.3)
            elif ch == 'x':
                vx, vy, vyaw = 0.0, 0.0, 0.0
            else:
                continue

            cmd = f"[{vx:.2f}, {vy:.2f}, {vyaw:.2f}, {height:.2f}]"
            pub.Write(String_(data=cmd))
            print(f"\r  vx={vx:.1f} vy={vy:.1f} yaw={vyaw:.1f} h={height:.2f}  ", end="", flush=True)

    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        pub.Write(String_(data="[0.0, 0.0, 0.0, 0.8]"))


if __name__ == "__main__":
    main()
