"""
Device Manager
==============
Selects the best available compute device (Intel GPU → NPU → CPU)
and provides helpers for model placement and compilation.

Supports:
  - OpenVINO GPU (Intel iGPU / dGPU)
  - OpenVINO NPU (Intel Panther Lake / Meteor Lake)
  - CUDA GPU (NVIDIA)
  - CPU fallback
"""

import logging
import os
from enum import Enum, auto
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class DeviceType(Enum):
    OPENVINO_GPU = auto()
    OPENVINO_NPU = auto()
    CUDA = auto()
    CPU = auto()


_OV_DEVICE_NAMES: dict = {}


def detect_best_device() -> DeviceType:
    """Probe available accelerators and return the best one."""
    if torch.cuda.is_available():
        logger.info("CUDA GPU detected: %s", torch.cuda.get_device_name(0))
        return DeviceType.CUDA

    try:
        from openvino import Core
        core = Core()
        devs = core.available_devices
        if "NPU" in devs:
            name = core.get_property("NPU", "FULL_DEVICE_NAME")
            _OV_DEVICE_NAMES["NPU"] = name
            logger.info("OpenVINO NPU detected: %s", name)
            return DeviceType.OPENVINO_NPU
        if "GPU" in devs:
            name = core.get_property("GPU", "FULL_DEVICE_NAME")
            _OV_DEVICE_NAMES["GPU"] = name
            logger.info("OpenVINO GPU detected: %s", name)
            return DeviceType.OPENVINO_GPU
    except ImportError:
        pass

    logger.info("No accelerator found — using CPU")
    return DeviceType.CPU


def get_torch_device(dev: DeviceType) -> torch.device:
    """Map DeviceType to a torch.device for tensor placement.

    OpenVINO devices keep tensors on CPU; the OpenVINO backend
    transparently moves them to the accelerator during inference.
    """
    if dev == DeviceType.CUDA:
        return torch.device("cuda")
    return torch.device("cpu")


def get_torch_dtype(dev: DeviceType) -> torch.dtype:
    """Choose optimal dtype for the device."""
    if dev == DeviceType.CUDA:
        return torch.bfloat16
    return torch.float32


def get_openvino_device_str(dev: DeviceType) -> str:
    """Map DeviceType to OpenVINO device string."""
    if dev == DeviceType.OPENVINO_NPU:
        return "NPU"
    if dev == DeviceType.OPENVINO_GPU:
        return "GPU"
    return "CPU"


def compile_for_device(model: torch.nn.Module, dev: DeviceType) -> torch.nn.Module:
    """Wrap model with torch.compile using the best backend for the device.

    For OpenVINO devices, passes the target device (GPU/NPU) so that
    inference runs on the accelerator, not CPU.
    """
    if dev in (DeviceType.OPENVINO_GPU, DeviceType.OPENVINO_NPU):
        ov_dev = get_openvino_device_str(dev)
        try:
            import openvino.torch  # noqa: F401
            compiled = torch.compile(
                model,
                backend="openvino",
                options={"device": ov_dev},
            )
            hw_name = _OV_DEVICE_NAMES.get(ov_dev, ov_dev)
            logger.info("Model compiled with OpenVINO backend → %s (%s)", ov_dev, hw_name)
            return compiled
        except Exception as e:
            logger.warning("OpenVINO GPU compile failed (%s), trying CPU", e)

    if dev == DeviceType.CPU:
        try:
            import openvino.torch  # noqa: F401
            compiled = torch.compile(
                model,
                backend="openvino",
                options={"device": "CPU"},
            )
            logger.info("Model compiled with OpenVINO backend → CPU")
            return compiled
        except Exception as e:
            logger.warning("OpenVINO CPU compile failed (%s), using eager", e)
            return model

    if dev == DeviceType.CUDA:
        try:
            compiled = torch.compile(model, mode="reduce-overhead")
            logger.info("Model compiled with inductor backend (CUDA)")
            return compiled
        except Exception as e:
            logger.warning("torch.compile failed (%s), using eager", e)

    return model
