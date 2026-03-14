# Inference Guide

## CPU Inference (default)

```python
import onnxruntime as ort

session = ort.InferenceSession("policies/g1_stand.onnx",
                               providers=["CPUExecutionProvider"])
action = session.run(None, {"obs": obs_array})[0]
```

## GPU Inference (Intel Arc via OpenVINO)

```bash
pip install openvino onnxruntime-openvino
```

```python
import onnxruntime as ort

session = ort.InferenceSession(
    "policies/g1_stand.onnx",
    providers=["OpenVINOExecutionProvider"],
    provider_options=[{"device_type": "GPU"}],
)
action = session.run(None, {"obs": obs_array})[0]
```

## NPU Inference (Intel NPU)

Requires: Intel NPU driver + firmware installed.

```python
session = ort.InferenceSession(
    "policies/g1_stand.onnx",
    providers=["OpenVINOExecutionProvider"],
    provider_options=[{"device_type": "NPU"}],
)
```

Check available devices:

```python
from openvino import Core
print(Core().available_devices)  # e.g. ['CPU', 'GPU', 'NPU']
```

## Observation Format

103 floats: `[linvel_local(3), gyro(3), gravity(3), command(3), joint_pos-default(29), joint_vel(29), last_action(29), phase(4)]`

## Action Format

29 floats in `[-1, 1]`: joint position offsets scaled by `ACTION_SCALE` (0.25).

## Policy Rate

50 Hz (20ms per inference step).
