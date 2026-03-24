"""
VLA Inference Engine
====================
Loads a Vision-Language-Action model (pi0.5 or OpenVLA) and runs inference
on the Intel GPU / NPU via OpenVINO, or CUDA, or CPU fallback.

Architecture
------------
1.  Camera image + language instruction → VLA model → action chunk
2.  Action chunk is 50 time-steps of 32-dim vectors (pi0.5)
3.  ActionMapper translates raw actions → G1 velocity or joint targets
4.  Actions are queued and consumed at WBC frequency (50 Hz)

The engine runs in its own thread at 0.5–2 Hz (depending on hardware)
and pushes action chunks to a thread-safe queue.
"""

import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from cortex.peripheral.device_manager import (
    DeviceType,
    compile_for_device,
    detect_best_device,
    get_torch_device,
    get_torch_dtype,
)

logger = logging.getLogger(__name__)

MODEL_REGISTRY = {
    "pi05": {
        "repo": "lerobot/pi05_base",
        "local": "models/pi05_base",
        "type": "pi05",
        "chunk_size": 50,
        "action_dim": 32,
        "image_keys": [
            "observation.images.base_0_rgb",
            "observation.images.left_wrist_0_rgb",
            "observation.images.right_wrist_0_rgb",
        ],
    },
    "openvla": {
        "repo": "openvla/openvla-7b",
        "local": "models/openvla-7b",
        "type": "openvla",
        "chunk_size": 1,
        "action_dim": 7,
        "image_keys": ["image"],
    },
}


@dataclass
class VLAOutput:
    """Raw VLA model output."""
    actions: np.ndarray          # (chunk_size, action_dim)
    instruction: str
    timestamp: float
    inference_time_ms: float
    device_used: str


class VLAEngine:
    """Loads and runs a VLA model for robot action prediction.

    Supports pi0.5 (via LeRobot) and OpenVLA (via transformers).
    Automatically selects the best compute device.
    """

    def __init__(
        self,
        model_name: str = "pi05",
        model_path: Optional[str] = None,
        device_override: Optional[str] = None,
        compile_model: bool = True,
    ):
        self.model_name = model_name
        self.model_info = MODEL_REGISTRY[model_name]
        self._model = None
        self._tokenizer = None
        self._loaded = False
        self._compile = compile_model

        # Device selection
        if device_override:
            self._device_type = DeviceType[device_override.upper()]
        else:
            self._device_type = detect_best_device()
        self._torch_device = get_torch_device(self._device_type)
        self._dtype = get_torch_dtype(self._device_type)

        # Model path
        project_root = Path(__file__).resolve().parents[2]
        if model_path:
            self._model_path = Path(model_path)
        else:
            self._model_path = project_root / self.model_info["local"]

        # Action queue (thread-safe)
        self._action_queue: deque = deque(maxlen=200)
        self._lock = threading.Lock()
        self._current_instruction: str = ""

        # Inference thread
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._new_instruction_event = threading.Event()

        # Stats
        self.last_inference_time_ms: float = 0
        self.total_inferences: int = 0

        logger.info(
            "VLAEngine created: model=%s  device=%s  path=%s",
            model_name, self._device_type.name, self._model_path,
        )

    # ── Model Loading ───────────────────────────────────────────────────────

    def load(self) -> bool:
        """Load the model. Returns True on success."""
        if self._loaded:
            return True

        if not self._model_path.exists():
            logger.error("Model not found at %s. Download with: "
                         "huggingface-cli download %s --local-dir %s",
                         self._model_path, self.model_info["repo"], self._model_path)
            return False

        logger.info("Loading VLA model '%s' from %s ...", self.model_name, self._model_path)
        t0 = time.perf_counter()

        try:
            if self.model_info["type"] == "pi05":
                self._load_pi05()
            elif self.model_info["type"] == "openvla":
                self._load_openvla()
            else:
                raise ValueError(f"Unknown model type: {self.model_info['type']}")
        except Exception as e:
            logger.error("Failed to load model: %s", e)
            return False

        elapsed = time.perf_counter() - t0
        logger.info("VLA model loaded in %.1fs on %s", elapsed, self._device_type.name)
        self._loaded = True
        return True

    def _load_pi05(self) -> None:
        from lerobot.policies.pi05 import PI05Policy
        from cortex.peripheral.device_manager import get_openvino_device_str

        # Override LeRobot's config.json device field before loading.
        # The saved config says "mps" (from training on Mac) — replace with "cpu"
        # so LeRobot doesn't warn.  The actual acceleration is handled by
        # torch.compile(backend='openvino', options={'device': 'GPU'}).
        import json
        cfg_path = self._model_path / "config.json"
        if cfg_path.exists():
            with open(cfg_path) as f:
                cfg = json.load(f)
            if cfg.get("device") not in ("cpu", None):
                logger.info("Overriding saved device '%s' → 'cpu' (OpenVINO handles GPU offload)",
                            cfg.get("device"))
                cfg["device"] = "cpu"
                with open(cfg_path, "w") as f:
                    json.dump(cfg, f, indent=2)

        self._model = PI05Policy.from_pretrained(str(self._model_path))

        # Patch vision encoder + attention masks for transformers >=4.50
        self._patch_embed_image()

        self._model.eval()
        self._model = self._model.to(device=self._torch_device, dtype=self._dtype)

        # Compile with OpenVINO targeting the accelerator (GPU/NPU)
        if self._compile:
            try:
                self._model = compile_for_device(self._model, self._device_type)
            except Exception as e:
                logger.warning("Compilation failed, using eager: %s", e)

        ov_dev = get_openvino_device_str(self._device_type)
        logger.info("PI05 policy ready — tensors on %s, inference via OpenVINO → %s",
                     self._torch_device, ov_dev)

        from transformers import AutoTokenizer
        try:
            self._tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")
        except Exception:
            logger.info("PaliGemma tokenizer gated — using Gemma tokenizer (compatible)")
            self._tokenizer = AutoTokenizer.from_pretrained("unsloth/gemma-2-2b")

    def _patch_embed_image(self) -> None:
        """Fix PaliGemma API change: newer transformers returns a plain tensor
        from get_image_features instead of an object with .pooler_output."""
        import types

        inner = self._model.model  # PI05Pytorch
        pg = inner.paligemma_with_expert

        original_embed = pg.embed_image

        def patched_embed_image(self_pg, image):
            out_dtype = image.dtype
            if image.dtype != torch.float32:
                image = image.to(torch.float32)
            image_outputs = self_pg.paligemma.model.get_image_features(image)
            hidden_size = self_pg.paligemma.config.text_config.hidden_size
            if isinstance(image_outputs, torch.Tensor):
                # transformers >=4.50: returns projected features / sqrt(hidden_size)
                features = image_outputs * (hidden_size ** 0.5)
            else:
                features = image_outputs.pooler_output * (hidden_size ** 0.5)
            if features.dtype != out_dtype:
                features = features.to(out_dtype)
            return features

        pg.embed_image = types.MethodType(patched_embed_image, pg)

        # Patch _prepare_attention_masks_4d: ensure boolean dtype
        original_prepare = inner._prepare_attention_masks_4d

        def patched_prepare(self_model, att_2d_masks):
            att_4d = att_2d_masks[:, None, :, :]
            from lerobot.policies.pi05.modeling_pi05 import OPENPI_ATTENTION_MASK_VALUE
            return torch.where(att_4d.bool(), 0.0, OPENPI_ATTENTION_MASK_VALUE)

        inner._prepare_attention_masks_4d = types.MethodType(patched_prepare, inner)
        logger.info("Patched embed_image + attention masks for transformers compatibility")

    def _load_openvla(self) -> None:
        from transformers import AutoModelForVision2Seq, AutoProcessor

        self._model = AutoModelForVision2Seq.from_pretrained(
            str(self._model_path),
            torch_dtype=self._dtype,
            device_map=str(self._torch_device),
        )
        self._model.eval()
        self._tokenizer = AutoProcessor.from_pretrained(str(self._model_path))

    # ── Inference ───────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(
        self,
        image: np.ndarray,
        instruction: str,
        state: Optional[np.ndarray] = None,
    ) -> VLAOutput:
        """Run a single VLA inference.

        Args:
            image: Camera frame (H, W, 3) uint8 or (H, W, 3) float [0,1]
            instruction: Natural language instruction
            state: Robot joint state vector (up to 32 dims, zero-padded)

        Returns:
            VLAOutput with action chunk
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        t0 = time.perf_counter()

        if self.model_info["type"] == "pi05":
            actions = self._predict_pi05(image, instruction, state)
        elif self.model_info["type"] == "openvla":
            actions = self._predict_openvla(image, instruction)
        else:
            raise ValueError(f"Unknown model type: {self.model_info['type']}")

        elapsed_ms = (time.perf_counter() - t0) * 1000
        self.last_inference_time_ms = elapsed_ms
        self.total_inferences += 1

        output = VLAOutput(
            actions=actions,
            instruction=instruction,
            timestamp=time.time(),
            inference_time_ms=elapsed_ms,
            device_used=self._device_type.name,
        )

        logger.info(
            "VLA inference #%d: %.0fms  chunk=%s  device=%s",
            self.total_inferences, elapsed_ms, actions.shape, self._device_type.name,
        )
        return output

    def _predict_pi05(
        self, image: np.ndarray, instruction: str, state: Optional[np.ndarray],
    ) -> np.ndarray:
        """Run pi0.5 inference."""
        batch = self._build_pi05_batch(image, instruction, state)
        action_chunk = self._model.predict_action_chunk(batch)
        # action_chunk shape: (1, chunk_size, action_dim) or (chunk_size, action_dim)
        actions = action_chunk.cpu().numpy()
        if actions.ndim == 3:
            actions = actions[0]
        return actions

    def _build_pi05_batch(
        self, image: np.ndarray, instruction: str, state: Optional[np.ndarray],
    ) -> Dict[str, torch.Tensor]:
        """Build the input batch dict for pi0.5."""
        # Image preprocessing: (H,W,3) uint8 → (1,3,224,224) float [0,1]
        if image.dtype == np.uint8:
            img_f = image.astype(np.float32) / 255.0
        else:
            img_f = image.astype(np.float32)

        from PIL import Image as PILImage
        pil = PILImage.fromarray((img_f * 255).astype(np.uint8) if img_f.max() <= 1.0 else img_f.astype(np.uint8))
        pil = pil.resize((224, 224))
        img_t = torch.from_numpy(np.array(pil)).float() / 255.0
        img_t = img_t.permute(2, 0, 1).unsqueeze(0)  # (1, 3, 224, 224)
        img_t = img_t.to(device=self._torch_device, dtype=self._dtype)

        # Create dummy images for missing cameras
        blank = torch.zeros_like(img_t)

        # Tokenize instruction
        tokens = self._tokenizer(
            instruction,
            return_tensors="pt",
            padding="max_length",
            max_length=200,
            truncation=True,
        )
        lang_tokens = tokens["input_ids"].to(self._torch_device)
        lang_masks = tokens["attention_mask"].to(device=self._torch_device, dtype=torch.bool)

        # State vector (pad to 32 dims)
        if state is not None:
            s = np.zeros(32, dtype=np.float32)
            s[:min(len(state), 32)] = state[:32]
        else:
            s = np.zeros(32, dtype=np.float32)
        state_t = torch.from_numpy(s).unsqueeze(0).to(device=self._torch_device, dtype=self._dtype)

        batch = {
            "observation.images.base_0_rgb": img_t,
            "observation.images.left_wrist_0_rgb": blank,
            "observation.images.right_wrist_0_rgb": blank,
            "observation.state": state_t,
            "observation.language.tokens": lang_tokens,
            "observation.language.attention_mask": lang_masks,
        }
        return batch

    def _predict_openvla(self, image: np.ndarray, instruction: str) -> np.ndarray:
        """Run OpenVLA inference."""
        from PIL import Image as PILImage
        pil = PILImage.fromarray(image)
        inputs = self._tokenizer(
            f"In: What action should the robot take to {instruction}?\nOut:",
            images=pil,
            return_tensors="pt",
        ).to(self._torch_device)

        out = self._model.generate(**inputs, max_new_tokens=256, do_sample=False)
        raw = self._tokenizer.decode(out[0], skip_special_tokens=True)

        # OpenVLA outputs 7 continuous values as text
        import re
        nums = re.findall(r"[-+]?\d*\.?\d+", raw)
        actions = np.array([float(x) for x in nums[:7]], dtype=np.float32)
        return actions.reshape(1, -1)

    # ── Background Inference Thread ─────────────────────────────────────────

    def start_background(self, get_image_fn, get_state_fn) -> None:
        """Start background inference loop.

        get_image_fn: callable returning (H,W,3) uint8 ndarray
        get_state_fn: callable returning 1D ndarray of joint positions
        """
        if self._running:
            return
        self._running = True
        self._get_image = get_image_fn
        self._get_state = get_state_fn
        self._thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._thread.start()
        logger.info("VLA background inference started")

    def stop_background(self) -> None:
        self._running = False
        self._new_instruction_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("VLA background inference stopped")

    def set_instruction(self, instruction: str) -> None:
        """Set a new language instruction for the VLA to execute."""
        with self._lock:
            self._current_instruction = instruction
            self._action_queue.clear()
        self._new_instruction_event.set()
        logger.info("VLA instruction set: %s", instruction)

    def get_next_action(self) -> Optional[np.ndarray]:
        """Pop the next action from the queue. Returns None if empty."""
        with self._lock:
            if self._action_queue:
                return self._action_queue.popleft()
        return None

    @property
    def has_actions(self) -> bool:
        with self._lock:
            return len(self._action_queue) > 0

    @property
    def queue_size(self) -> int:
        with self._lock:
            return len(self._action_queue)

    def _inference_loop(self) -> None:
        """Background loop: waits for instructions, runs inference, queues actions."""
        while self._running:
            self._new_instruction_event.wait(timeout=1.0)
            self._new_instruction_event.clear()

            with self._lock:
                instruction = self._current_instruction

            if not instruction or not self._running:
                continue

            try:
                image = self._get_image()
                state = self._get_state()
                if image is None:
                    continue

                output = self.predict(image, instruction, state)

                with self._lock:
                    for i in range(output.actions.shape[0]):
                        self._action_queue.append(output.actions[i])

            except Exception as e:
                logger.error("VLA inference error: %s", e)
                time.sleep(1.0)

    # ── Properties ──────────────────────────────────────────────────────────

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def device_type(self) -> DeviceType:
        return self._device_type

    @property
    def model_path(self) -> Path:
        return self._model_path

    def close(self) -> None:
        self.stop_background()
        self._model = None
        self._loaded = False
        logger.info("VLAEngine closed")
