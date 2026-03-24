"""
VLM Engine
==========
Loads PaliGemma 3B (or a fallback VLM) and provides:
  - caption(image, prompt) → text description of the image
  - generate_text(prompt)  → text generation from the Gemma 2B component

Uses the existing device_manager for Intel iGPU / OpenVINO acceleration.
PaliGemma 3B = SigLIP vision encoder (400M) + Gemma 2B language model.
We reuse the language model component for planning text generation.
"""

import logging
import os
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from cortex.peripheral.device_manager import (
    DeviceType,
    compile_for_device,
    detect_best_device,
    get_openvino_device_str,
    get_torch_device,
    get_torch_dtype,
)

logger = logging.getLogger(__name__)

# Moondream decode uses a fixed 2048-position mask; long prompts + long gen overflow.
_MOONDREAM_PLAN_PROMPT_MAX_CHARS = 1600
_MOONDREAM_PLAN_MAX_NEW_TOKENS = 160

MODEL_REGISTRY = {
    "paligemma": {
        "repo": "google/paligemma-3b-pt-224",
        "local": "models/paligemma_3b",
        "image_size": 224,
        "type": "paligemma",
    },
    "paligemma3b": {
        "repo": "google/paligemma-3b-pt-224",
        "local": "/home/hf-datasets/paligemma_3b",
        "image_size": 224,
        "type": "paligemma",
        "abs_path": True,
    },
    "paligemma2": {
        "repo": "google/paligemma2-3b-pt-224",
        "local": "models/paligemma2_3b",
        "image_size": 224,
        "type": "paligemma",
    },
    "qwen2vl": {
        "repo": "Qwen/Qwen2-VL-2B-Instruct",
        "local": "models/qwen2vl_2b",
        "image_size": 448,
        "type": "qwen2vl",
    },
    "moondream": {
        "repo": "vikhyatk/moondream2",
        "local": "models/moondream2",
        "image_size": 378,
        "type": "moondream",
    },
    "smolvlm": {
        "repo": "HuggingFaceTB/SmolVLM-256M-Instruct",
        "local": "models/smolvlm_256m",
        "ov_local": "models/smolvlm_256m_ov",
        "image_size": 512,
        "type": "smolvlm",
    },
    "smolvlm500m": {
        "repo": "HuggingFaceTB/SmolVLM-500M-Instruct",
        "local": "models/smolvlm_500m",
        "ov_local": "models/smolvlm_500m_ov",
        "image_size": 512,
        "type": "smolvlm",
    },
}


class VLMEngine:
    """Vision-Language Model for scene captioning and text generation.

    Loads PaliGemma 3B and exposes two inference modes:
      1. Full VLM: image + prompt → text (for camera captioning)
      2. Text-only: prompt → text (for planning, reusing the Gemma 2B backbone)
    """

    def __init__(
        self,
        model_name: str = "paligemma",
        model_path: Optional[str] = None,
        device_override: Optional[str] = None,
        compile_model: bool = True,
        max_new_tokens: int = 256,
    ):
        self.model_name = model_name
        self.model_info = MODEL_REGISTRY.get(model_name, MODEL_REGISTRY["paligemma"])
        self._model = None
        self._processor = None
        self._loaded = False
        self._compile = compile_model
        self._max_new_tokens = max_new_tokens

        if device_override:
            self._device_type = DeviceType[device_override.upper()]
        else:
            self._device_type = detect_best_device()
        self._torch_device = get_torch_device(self._device_type)
        self._dtype = get_torch_dtype(self._device_type)

        project_root = Path(__file__).resolve().parents[2]
        if model_path:
            self._model_path = Path(model_path)
        elif self.model_info.get("abs_path"):
            self._model_path = Path(self.model_info["local"])
        else:
            self._model_path = project_root / self.model_info["local"]

        self.last_inference_ms: float = 0
        self.total_inferences: int = 0
        # Moondream: reuse dummy image embed for JSON planning; serialize GPU work
        # (planner thread + sim can otherwise hit the same OpenVINO/PyTorch stack).
        self._moondream_dummy_enc = None
        self._moondream_infer_lock = threading.Lock()

        logger.info(
            "VLMEngine created: model=%s device=%s path=%s",
            model_name, self._device_type.name, self._model_path,
        )

    @staticmethod
    def _truncate_moondream_planning_prompt(prompt: str) -> str:
        """Keep instructions + user message; drop middle of scene context if needed."""
        mark = "\n\n## User message (verbatim)\n"
        max_total = _MOONDREAM_PLAN_PROMPT_MAX_CHARS
        if mark in prompt:
            head, _, tail = prompt.partition(mark)
            tail = mark + tail
            max_head = max(400, max_total - len(tail))
            if len(head) > max_head:
                head = (
                    head[:max_head]
                    + "\n\n[Scene context truncated — Moondream 2048-token context limit.]\n"
                )
            prompt = head + tail
        elif len(prompt) > max_total:
            prompt = (
                prompt[: max_total - 40]
                + "\n\n[Truncated for model context limit.]\n"
            )
        return prompt

    def _moondream_generate_text(self, enc, prompt: str, max_new_tokens: int) -> str:
        """Run Moondream query with a safe prompt length and generation cap."""
        prompt = self._truncate_moondream_planning_prompt(prompt)
        max_gen = min(int(max_new_tokens), _MOONDREAM_PLAN_MAX_NEW_TOKENS)
        settings = {"max_tokens": max_gen, "temperature": 0.0}
        query_fn = getattr(self._model, "query", None)
        if callable(query_fn):
            try:
                out = query_fn(
                    image=enc, question=prompt, stream=False, settings=settings,
                )
                if isinstance(out, dict) and "answer" in out:
                    return str(out["answer"]).strip()
            except TypeError:
                pass
            except Exception as e:
                logger.warning("Moondream query(settings=) failed, falling back: %s", e)
        # Older wrappers: answer_question ignores max_new_tokens for query — still use short prompt.
        return str(
            self._model.answer_question(
                enc, prompt, self._processor, max_new_tokens=max_gen,
            )
        ).strip()

    def load(self) -> bool:
        """Download (if needed) and load the model. Returns True on success."""
        if self._loaded:
            return True

        if not self._model_path.exists():
            logger.info("Model not found locally, downloading from %s ...", self.model_info["repo"])
            if not self._download():
                return False

        logger.info("Loading VLM '%s' from %s ...", self.model_name, self._model_path)
        t0 = time.perf_counter()

        try:
            model_type = self.model_info.get("type", "paligemma")
            if model_type == "smolvlm":
                self._load_smolvlm()
            elif model_type == "qwen2vl":
                self._load_qwen2vl()
            elif model_type == "moondream":
                self._load_moondream()
            else:
                self._load_paligemma()
        except Exception as e:
            logger.error("Failed to load VLM: %s", e, exc_info=True)
            return False

        elapsed = time.perf_counter() - t0
        logger.info("VLM loaded in %.1fs on %s", elapsed, self._device_type.name)
        self._loaded = True
        return True

    def _download(self) -> bool:
        """Download model from HuggingFace Hub."""
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(
                self.model_info["repo"],
                local_dir=str(self._model_path),
                ignore_patterns=["*.bin", "*.gguf"],
            )
            logger.info("Downloaded %s to %s", self.model_info["repo"], self._model_path)
            return True
        except Exception as e:
            logger.error("Download failed: %s", e)
            return False

    def _load_paligemma(self) -> None:
        from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

        self._processor = AutoProcessor.from_pretrained(str(self._model_path))
        self._model = PaliGemmaForConditionalGeneration.from_pretrained(
            str(self._model_path),
            torch_dtype=self._dtype,
        )
        self._model.eval()
        self._model = self._model.to(self._torch_device)
        self._model_type = "paligemma"

        if self._compile:
            try:
                self._model = compile_for_device(self._model, self._device_type)
            except Exception as e:
                logger.warning("Compilation failed, using eager mode: %s", e)

    def _load_qwen2vl(self) -> None:
        from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, Qwen2VLImageProcessor

        self._tokenizer = AutoTokenizer.from_pretrained(str(self._model_path))
        self._image_processor = Qwen2VLImageProcessor.from_pretrained(str(self._model_path))
        self._processor = self._tokenizer  # for text-only calls
        self._model = Qwen2VLForConditionalGeneration.from_pretrained(
            str(self._model_path),
            dtype=self._dtype,
        )
        self._model.eval()
        self._model = self._model.to(self._torch_device)
        self._model_type = "qwen2vl"

        if self._compile:
            try:
                self._model = compile_for_device(self._model, self._device_type)
            except Exception as e:
                logger.warning("Compilation failed, using eager mode: %s", e)

    def _load_moondream(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import shutil

        # Moondream uses trust_remote_code; ensure custom modules are
        # in the HF cache where transformers expects them.
        cache_dir = Path.home() / ".cache/huggingface/modules/transformers_modules/moondream2"
        if not cache_dir.exists():
            cache_dir.mkdir(parents=True, exist_ok=True)
            for py_file in self._model_path.glob("*.py"):
                shutil.copy2(py_file, cache_dir / py_file.name)

        self._processor = AutoTokenizer.from_pretrained(
            str(self._model_path), trust_remote_code=True,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            str(self._model_path),
            dtype=self._dtype,
            trust_remote_code=True,
        )
        self._model.eval()
        self._model_type = "moondream"

    def _load_smolvlm(self) -> None:
        from transformers import AutoProcessor

        self._processor = AutoProcessor.from_pretrained(
            str(self._model_path),
            size={"longest_edge": 768},
        )

        if self._device_type == DeviceType.CUDA:
            from transformers import AutoModelForVision2Seq
            self._model = AutoModelForVision2Seq.from_pretrained(
                str(self._model_path), dtype=torch.bfloat16,
                _attn_implementation="eager",
            )
            self._model.eval().to("cuda")
            self._model_type = "smolvlm"
            logger.info("SmolVLM loaded on CUDA (bfloat16)")
            return

        # OpenVINO path: use pre-exported IR if available, else export on first run
        from optimum.intel import OVModelForVisualCausalLM

        project_root = Path(__file__).resolve().parents[2]
        ov_rel = self.model_info.get("ov_local")
        ov_path = project_root / ov_rel if ov_rel else None

        if ov_path and ov_path.exists() and (ov_path / "openvino_language_model.xml").exists():
            logger.info("Loading pre-exported OpenVINO IR from %s", ov_path)
            ov_device = get_openvino_device_str(self._device_type)
            self._model = OVModelForVisualCausalLM.from_pretrained(
                str(ov_path), device=ov_device,
            )
        else:
            logger.info("No pre-exported IR found — exporting SmolVLM to OpenVINO (one-time, ~25s)...")
            self._model = OVModelForVisualCausalLM.from_pretrained(
                str(self._model_path), export=True, device="CPU",
            )
            if ov_path:
                self._model.save_pretrained(str(ov_path))
                logger.info("OpenVINO IR saved to %s for fast reload", ov_path)
            ov_device = get_openvino_device_str(self._device_type)
            if ov_device != "CPU":
                self._model = OVModelForVisualCausalLM.from_pretrained(
                    str(ov_path or self._model_path), device=ov_device,
                )

        self._model_type = "smolvlm_ov"
        logger.info("SmolVLM loaded via OpenVINO on %s (img=768, 5 patches)",
                     get_openvino_device_str(self._device_type))

    @torch.no_grad()
    def caption(self, image: np.ndarray, prompt: str = "Describe what you see.") -> str:
        """Run full VLM inference: image + prompt → text description."""
        if not self._loaded:
            raise RuntimeError("VLM not loaded. Call load() first.")

        t0 = time.perf_counter()
        from PIL import Image as PILImage

        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        pil_img = PILImage.fromarray(image)

        model_type = getattr(self, "_model_type", "paligemma")

        if model_type in ("smolvlm", "smolvlm_ov"):
            text = self._caption_smolvlm(pil_img, prompt)
        elif model_type == "qwen2vl":
            text = self._caption_qwen2vl(pil_img, prompt)
        elif model_type == "moondream":
            text = self._caption_moondream(pil_img, prompt)
        else:
            text = self._caption_paligemma(pil_img, prompt)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        self.last_inference_ms = elapsed_ms
        self.total_inferences += 1
        logger.info("VLM caption #%d: %.0fms  prompt='%s'  output_len=%d",
                     self.total_inferences, elapsed_ms, prompt[:50], len(text))
        return text.strip()

    def _caption_paligemma(self, pil_img, prompt: str) -> str:
        inputs = self._processor(
            text="<image>" + prompt, images=pil_img, return_tensors="pt",
        ).to(device=self._torch_device, dtype=self._dtype)
        output_ids = self._model.generate(**inputs, max_new_tokens=self._max_new_tokens, do_sample=False)
        input_len = inputs["input_ids"].shape[1]
        return self._processor.decode(output_ids[0][input_len:], skip_special_tokens=True)

    def _caption_qwen2vl(self, pil_img, prompt: str) -> str:
        # Use image processor + tokenizer separately (avoids torchvision dependency)
        image_inputs = self._image_processor(images=[pil_img], return_tensors="pt")
        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt},
        ]}]
        text_input = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        text_inputs = self._tokenizer(text_input, return_tensors="pt")
        inputs = {**text_inputs, **image_inputs}
        inputs = {k: v.to(self._torch_device) for k, v in inputs.items()}
        output_ids = self._model.generate(**inputs, max_new_tokens=self._max_new_tokens, do_sample=False)
        input_len = text_inputs["input_ids"].shape[1]
        return self._tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True)

    def _caption_moondream(self, pil_img, prompt: str) -> str:
        with self._moondream_infer_lock:
            enc_image = self._model.encode_image(pil_img)
            return self._model.answer_question(enc_image, prompt, self._processor)

    def _caption_smolvlm(self, pil_img, prompt: str) -> str:
        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt},
        ]}]
        text_input = self._processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self._processor(text=text_input, images=[pil_img], return_tensors="pt")
        inputs = {k: v.to(self._torch_device) for k, v in inputs.items()}
        cap_tokens = min(self._max_new_tokens, 128)
        output_ids = self._model.generate(**inputs, max_new_tokens=cap_tokens, do_sample=False)
        input_len = inputs["input_ids"].shape[1]
        return self._processor.batch_decode(output_ids[:, input_len:], skip_special_tokens=True)[0]

    @torch.no_grad()
    def generate_text(self, prompt: str, max_new_tokens: Optional[int] = None) -> str:
        """Text-only generation using the VLM's language backbone.

        Used by the Planner for JSON task-chain planning.  Feeds a dummy image
        for models that require one (PaliGemma, Moondream).
        """
        if not self._loaded:
            raise RuntimeError("VLM not loaded. Call load() first.")

        max_tok = int(max_new_tokens) if max_new_tokens is not None else self._max_new_tokens

        t0 = time.perf_counter()
        from PIL import Image as PILImage

        model_type = getattr(self, "_model_type", "paligemma")
        img_size = self.model_info.get("image_size", 224)
        dummy_img = PILImage.new("RGB", (img_size, img_size), (128, 128, 128))

        if model_type in ("smolvlm", "smolvlm_ov"):
            text = self._generate_text_smolvlm(prompt, max_new_tokens=max_tok)
        elif model_type == "qwen2vl":
            text = self._generate_text_qwen2vl(prompt, max_new_tokens=max_tok)
        elif model_type == "moondream":
            enc = self._model.encode_image(dummy_img)
            text = self._moondream_generate_text(enc, prompt, max_tok)
        else:
            inputs = self._processor(
                text="<image>" + prompt, images=dummy_img, return_tensors="pt",
            ).to(device=self._torch_device, dtype=self._dtype)
            output_ids = self._model.generate(
                **inputs, max_new_tokens=max_tok, do_sample=False,
            )
            input_len = inputs["input_ids"].shape[1]
            text = self._processor.decode(output_ids[0][input_len:], skip_special_tokens=True)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        self.last_inference_ms = elapsed_ms
        self.total_inferences += 1
        logger.info("VLM text gen #%d: %.0fms  prompt_len=%d  output_len=%d",
                     self.total_inferences, elapsed_ms, len(prompt), len(text))
        return text.strip()

    def _generate_text_qwen2vl(self, prompt: str, max_new_tokens: Optional[int] = None) -> str:
        """Qwen2-VL supports text-only input natively."""
        max_tok = int(max_new_tokens) if max_new_tokens is not None else self._max_new_tokens
        messages = [{"role": "user", "content": prompt}]
        text_input = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._tokenizer(text_input, return_tensors="pt").to(self._torch_device)
        output_ids = self._model.generate(**inputs, max_new_tokens=max_tok, do_sample=False)
        input_len = inputs["input_ids"].shape[1]
        return self._tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True)

    def _generate_text_smolvlm(self, prompt: str, max_new_tokens: Optional[int] = None) -> str:
        """SmolVLM text-only: feed a tiny dummy image so the vision encoder is happy."""
        max_tok = int(max_new_tokens) if max_new_tokens is not None else self._max_new_tokens
        from PIL import Image as PILImage
        dummy = PILImage.new("RGB", (64, 64), (128, 128, 128))
        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt},
        ]}]
        text_input = self._processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self._processor(text=text_input, images=[dummy], return_tensors="pt")
        inputs = {k: v.to(self._torch_device) for k, v in inputs.items()}
        output_ids = self._model.generate(**inputs, max_new_tokens=max_tok, do_sample=False)
        input_len = inputs["input_ids"].shape[1]
        return self._processor.batch_decode(output_ids[:, input_len:], skip_special_tokens=True)[0]

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def device_type(self) -> DeviceType:
        return self._device_type

    def close(self) -> None:
        self._moondream_dummy_enc = None
        self._model = None
        self._processor = None
        self._loaded = False
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        logger.info("VLMEngine closed")
