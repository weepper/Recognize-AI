"""
Model loading and ONNX session management.

Models are expected to be pre-downloaded into MODELS_DIR by running download_models.py.
"""
import os
import logging
from typing import Optional

import onnxruntime as ort

from config import (
    MODELS_DIR,
    MODEL_NAMES,
    ONNX_PROVIDERS,
    USE_QUANTIZED,
    INTRA_OP_THREADS,
    INTER_OP_THREADS,
    EXECUTION_MODE,
)

logger = logging.getLogger(__name__)


def find_model(model_name: str) -> str:
    """Find a model file in MODELS_DIR. Returns the path or raises FileNotFoundError."""
    if USE_QUANTIZED:
        quant_path = os.path.join(MODELS_DIR, f"{model_name}.quant.onnx")
        if os.path.exists(quant_path):
            logger.info(f"Using quantized model variant: {os.path.basename(quant_path)}")
            return quant_path

    model_path = os.path.join(MODELS_DIR, f"{model_name}.onnx")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model '{model_name}' not found (neither standard nor quantized). "
            f"Run 'python download_models.py' to download all models."
        )
    return model_path


def create_inference_session(model_path: str) -> ort.InferenceSession:
    """Create an ONNX Runtime InferenceSession with the configured providers and options."""
    available = ort.get_available_providers()
    providers = [p.strip() for p in ONNX_PROVIDERS if p.strip() in available]
    if not providers:
        providers = ["CPUExecutionProvider"]

    # Configure session options
    opts = ort.SessionOptions()
    if INTRA_OP_THREADS is not None:
        opts.intra_op_num_threads = INTRA_OP_THREADS
        logger.info(f"Setting intra_op_num_threads = {INTRA_OP_THREADS}")
    if INTER_OP_THREADS is not None:
        opts.inter_op_num_threads = INTER_OP_THREADS
        logger.info(f"Setting inter_op_num_threads = {INTER_OP_THREADS}")

    if EXECUTION_MODE == "parallel":
        opts.execution_mode = ort.ExecutionMode.ORT_PARALLEL
        logger.info("Setting execution_mode = ORT_PARALLEL")
    else:
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        logger.info("Setting execution_mode = ORT_SEQUENTIAL")

    logger.info(f"Creating session for {os.path.basename(model_path)} | using={providers}")
    session = ort.InferenceSession(model_path, sess_options=opts, providers=providers)
    logger.info(f"Active providers: {session.get_providers()}")
    return session


class ModelManager:
    """Lazy-loading singleton that keeps ONNX sessions alive."""

    def __init__(self):
        self._sessions: dict[str, Optional[ort.InferenceSession]] = {}
        self._loaded: dict[str, bool] = {}
        self._errors: dict[str, str] = {}

    def load(self, name: str) -> bool:
        """Load a single model. Returns True on success."""
        if name in self._sessions and self._sessions[name] is not None:
            return True
        try:
            path = find_model(name)
            self._sessions[name] = create_inference_session(path)
            self._loaded[name] = True
            self._errors.pop(name, None)
            logger.info(f"Model '{name}' loaded successfully.")
            return True
        except Exception as e:
            self._sessions[name] = None
            self._loaded[name] = False
            self._errors[name] = str(e)
            logger.warning(f"Failed to load model '{name}': {e}")
            return False

    def load_all(self):
        """Attempt to load every known model."""
        for name in MODEL_NAMES:
            self.load(name)

    def get(self, name: str) -> Optional[ort.InferenceSession]:
        return self._sessions.get(name)

    def status(self) -> dict:
        """Return a dict describing which models are loaded, with metadata."""
        result = {}
        for name in MODEL_NAMES:
            info = {
                "loaded": self._loaded.get(name, False),
                "error": self._errors.get(name),
                "quantized": False,
                "path": None,
            }
            # Determine if the loaded model is quantized
            session = self._sessions.get(name)
            if session is not None:
                try:
                    model_path = session._model_path if hasattr(session, '_model_path') else None
                    if model_path and ".quant." in model_path:
                        info["quantized"] = True
                    info["path"] = model_path
                except Exception:
                    pass
                # Try to detect from the find_model path
                if not info["path"]:
                    try:
                        path = find_model(name)
                        info["quantized"] = ".quant." in path
                        info["path"] = path
                    except FileNotFoundError:
                        pass
            result[name] = info
        return result

    def get_gpu_info(self) -> dict:
        """Query ONNX Runtime for GPU/execution provider information."""
        available_providers = ort.get_available_providers()
        active_providers = []

        # Check what providers are actually being used by loaded sessions
        for name, session in self._sessions.items():
            if session is not None:
                active_providers = list(session.get_providers())
                break

        gpu_info = {
            "available_providers": available_providers,
            "active_providers": active_providers if active_providers else ["CPUExecutionProvider"],
            "has_gpu": "CUDAExecutionProvider" in available_providers,
            "using_gpu": "CUDAExecutionProvider" in (active_providers or []),
            "device": None,
            "cuda_version": None,
        }

        # Try to get CUDA device info
        if gpu_info["has_gpu"]:
            try:
                device_info = ort.get_device()
                gpu_info["device"] = device_info
            except Exception:
                pass

        return gpu_info


# Global singleton
model_manager = ModelManager()


class TokenizerManager:
    """Lazy-loading manager for the HuggingFace CLIP Tokenizer."""

    def __init__(self):
        self._tokenizer = None

    def get(self):
        if self._tokenizer is None:
            from tokenizers import Tokenizer
            from config import CLIP_TOKENIZER
            logger.info(f"Loading CLIP Tokenizer from {CLIP_TOKENIZER} ...")
            self._tokenizer = Tokenizer.from_pretrained(CLIP_TOKENIZER)
            # CLIP sequence context length is exactly 77 tokens
            self._tokenizer.enable_padding(length=77)
            self._tokenizer.enable_truncation(max_length=77)
            logger.info("CLIP Tokenizer loaded successfully.")
        return self._tokenizer


tokenizer_manager = TokenizerManager()

