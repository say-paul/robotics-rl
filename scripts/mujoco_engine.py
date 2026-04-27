"""Backward compatibility — implementation moved to engines/mujoco_engine.py."""
from engines.mujoco_engine import (          # noqa: F401
    run_simulation,
    MuJoCoEngine,
    VirtualHarness,
    pd_control,
)
