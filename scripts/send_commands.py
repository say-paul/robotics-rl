#!/usr/bin/env python3
# This project was developed with assistance from AI tools.
"""Keyboard velocity command sender for unitree_sim_isaaclab.

Publishes velocity commands to rt/run_command/cmd as string-encoded
lists [vx, vy, yaw_rate, height] over CycloneDDS. The sim's
action_provider_wh_dds.py reads these to control the robot.

Usage:
    python scripts/send_commands.py
"""

import sys
import tty
import termios
import time

from cyclonedds.domain import DomainParticipant
from cyclonedds.pub import DataWriter
from cyclonedds.topic import Topic
from cyclonedds.idl import IdlStruct
from dataclasses import dataclass
import cyclonedds.idl.types as types


@dataclass
class StringMsg(IdlStruct, typename="unitree_go.msg.dds_.String_"):
    data: str


def main():
    dp = DomainParticipant(domain_id=1)
    writer = DataWriter(dp, Topic(dp, "rt/run_command/cmd", StringMsg))

    vx, vy, vyaw, height = 0.0, 0.0, 0.0, 0.8

    print("G1 Velocity Commander")
    print("=" * 40)
    print("  w/s  — forward/back")
    print("  a/d  — turn left/right")
    print("  q/e  — strafe left/right")
    print("  r/f  — height up/down")
    print("  x    — stop all")
    print("  Ctrl-C — quit")
    print("=" * 40)

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
            writer.write(StringMsg(data=cmd))
            print(f"\r  vx={vx:.1f} vy={vy:.1f} yaw={vyaw:.1f} h={height:.2f}  ", end="", flush=True)

    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        writer.write(StringMsg(data="[0.0, 0.0, 0.0, 0.8]"))


if __name__ == "__main__":
    main()
