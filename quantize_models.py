"""
ONNX Model Quantization tool for Recognize AI Backend.
Converts FP32 models to dynamic INT8 precision to save size and speed up inference.
"""
import os
import sys
import logging
from time import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

try:
    from onnxruntime.quantization import quantize_dynamic, QuantType
except ImportError:
    logger.error("ONNX Runtime quantization tools not available. Please install optimum[onnxruntime] or onnxruntime.")
    sys.exit(1)


def quantize_model(model_name: str):
    """Perform dynamic dynamic activation/weight quantization."""
    src = os.path.join(MODELS_DIR, f"{model_name}.onnx")
    dest = os.path.join(MODELS_DIR, f"{model_name}.quant.onnx")

    if not os.path.exists(src):
        logger.warning(f"Source model not found: {src}. Skipping.")
        return

    logger.info(f"--- Quantizing {model_name} ---")
    size_before = os.path.getsize(src) / 1024 / 1024
    logger.info(f"Original size: {size_before:.2f} MB")

    t0 = time()
    try:
        quantize_dynamic(
            model_input=src,
            model_output=dest,
            weight_type=QuantType.QUInt8,
        )
        t1 = time()
        size_after = os.path.getsize(dest) / 1024 / 1024
        reduction = (1 - (size_after / size_before)) * 100
        logger.info(f"Quantized saved to: {dest}")
        logger.info(f"Quantized size: {size_after:.2f} MB ({reduction:.1f}% reduction)")
        logger.info(f"Completed in {t1 - t0:.2f} seconds\n")
    except Exception as e:
        logger.error(f"Failed to quantize {model_name}: {e}\n")


if __name__ == "__main__":
    models = sys.argv[1:] if len(sys.argv) > 1 else ["yolov8n", "arcface", "clip_visual"]
    for model in models:
        quantize_model(model)
    logger.info("Quantization process completed.")
