"""
Training Package
================
Tools for generating LeRobot-compatible datasets from bone-seed motion
data and the MuJoCo water_bottle_stage simulation, then training with
LeRobot's SARM and SmolVLA pipelines.

Modules:
  motion_library    – CSV bone-seed loader & clip index
  playback          – Motion playback visualiser
  generate_dataset  – LeRobot v3.0 dataset generator (bone-seed → images + states)
"""
