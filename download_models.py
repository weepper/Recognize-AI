"""
Download / export ONNX models for Recognize ExApp.

Usage:
    python download_models.py          # download all models
    python download_models.py yolo     # download only YOLOv8n
    python download_models.py arcface  # download only ArcFace
    python download_models.py clip     # export only CLIP visual encoder

Requirements (install via requirements-dev.txt):
    pip install -r requirements-dev.txt
"""
import os
import sys
import shutil
import subprocess
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

ARCFACE_URL = "https://huggingface.co/FoivosPar/Arc2Face/resolve/main/arcface.onnx"


def download_yolo():
    """Export YOLOv8n to ONNX using the ultralytics package."""
    dest = os.path.join(MODELS_DIR, "yolov8n.onnx")
    if os.path.exists(dest):
        logger.info(f"YOLOv8n already exists at {dest}")
        return

    logger.info("Exporting YOLOv8n to ONNX via ultralytics ...")
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.info("Installing ultralytics ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
        from ultralytics import YOLO

    model = YOLO("yolov8n.pt")  # auto-downloads the PyTorch model
    exported = model.export(format="onnx")  # exports and returns the path
    if exported and os.path.exists(exported):
        shutil.move(exported, dest)
        logger.info(f"YOLOv8n ONNX saved to {dest} ({os.path.getsize(dest) / 1024 / 1024:.1f} MB)")
    else:
        raise RuntimeError("ultralytics export failed – no ONNX file produced")


def download_arcface():
    """Download ArcFace ONNX from HuggingFace."""
    dest = os.path.join(MODELS_DIR, "arcface.onnx")
    if os.path.exists(dest):
        logger.info(f"ArcFace already exists at {dest}")
        return

    logger.info(f"Downloading ArcFace from {ARCFACE_URL} ...")
    try:
        import httpx
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "httpx"])
        import httpx

    headers = {"User-Agent": "recognize-exapp/1.0"}
    with httpx.stream("GET", ARCFACE_URL, follow_redirects=True, headers=headers, timeout=300) as resp:
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_bytes(chunk_size=8192):
                f.write(chunk)

    logger.info(f"ArcFace saved to {dest} ({os.path.getsize(dest) / 1024 / 1024:.1f} MB)")


def export_clip():
    """Export CLIP ViT-B/32 vision encoder to ONNX using HuggingFace optimum."""
    dest = os.path.join(MODELS_DIR, "clip_visual.onnx")
    if os.path.exists(dest):
        logger.info(f"CLIP visual model already exists at {dest}")
        return

    model_id = "openai/clip-vit-base-patch32"
    export_dir = os.path.join(os.path.dirname(__file__), "clip_export_api")

    logger.info(f"Exporting {model_id} to ONNX via optimum ...")
    try:
        from optimum.onnxruntime import ORTModelForFeatureExtraction
    except ImportError:
        logger.info("Installing optimum[onnxruntime] ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "optimum[onnxruntime]"])
        from optimum.onnxruntime import ORTModelForFeatureExtraction

    model = ORTModelForFeatureExtraction.from_pretrained(model_id, export=True)
    model.save_pretrained(export_dir)

    # The exported vision model may be named 'model.onnx' or 'vision_model.onnx'
    for candidate in ["vision_model.onnx", "model.onnx"]:
        src = os.path.join(export_dir, candidate)
        if os.path.exists(src):
            shutil.copy(src, dest)
            logger.info(f"CLIP visual model saved to {dest} ({os.path.getsize(dest) / 1024 / 1024:.1f} MB)")
            return

    raise RuntimeError(
        f"CLIP export completed but no ONNX file found in {export_dir}. "
        f"Files present: {os.listdir(export_dir)}"
    )


ALL_TASKS = {
    "yolo": download_yolo,
    "arcface": download_arcface,
    "clip": export_clip,
}

if __name__ == "__main__":
    tasks = sys.argv[1:] if len(sys.argv) > 1 else list(ALL_TASKS.keys())
    for task in tasks:
        if task in ALL_TASKS:
            ALL_TASKS[task]()
        else:
            logger.warning(f"Unknown task: {task}. Available: {', '.join(ALL_TASKS.keys())}")
    logger.info("Done!")
