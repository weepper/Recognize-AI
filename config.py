"""
Configuration via environment variables.

Supports dual-mode operation:
  - Standalone: python main.py (dev mode)
  - ExApp: Managed by Nextcloud AppAPI (APP_ID is set)
"""
import os

# ---------------------------------------------------------------------------
# App metadata
# ---------------------------------------------------------------------------
APP_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# ExApp detection — set by Nextcloud AppAPI at container startup
# ---------------------------------------------------------------------------
APP_ID = os.environ.get("APP_ID")
APP_SECRET = os.environ.get("APP_SECRET")
APP_PORT = os.environ.get("APP_PORT")
NEXTCLOUD_URL = os.environ.get("NEXTCLOUD_URL")

EXAPP_MODE = APP_ID is not None

# ---------------------------------------------------------------------------
# Model settings
# ---------------------------------------------------------------------------
# Directory where ONNX models are stored / downloaded to
MODELS_DIR = os.environ.get("RECOGNIZE_MODELS_DIR", os.path.join(os.path.dirname(__file__), "models"))

# Server settings (standalone mode only — ExApp mode uses APP_PORT)
HOST = os.environ.get("RECOGNIZE_HOST", "127.0.0.1")
PORT = int(os.environ.get("RECOGNIZE_PORT", APP_PORT or "8000"))

# Maximum upload file size in bytes (default 20 MB)
MAX_UPLOAD_SIZE = int(os.environ.get("RECOGNIZE_MAX_UPLOAD_SIZE", str(20 * 1024 * 1024)))

# ONNX Runtime execution providers (comma-separated)
# Options: CUDAExecutionProvider, TensorrtExecutionProvider, CPUExecutionProvider
ONNX_PROVIDERS = os.environ.get(
    "RECOGNIZE_ONNX_PROVIDERS",
    "CUDAExecutionProvider,CPUExecutionProvider"
).split(",")

# ONNX Session configuration options
INTRA_OP_THREADS = os.environ.get("RECOGNIZE_INTRA_OP_THREADS")
if INTRA_OP_THREADS is not None:
    INTRA_OP_THREADS = int(INTRA_OP_THREADS)

INTER_OP_THREADS = os.environ.get("RECOGNIZE_INTER_OP_THREADS")
if INTER_OP_THREADS is not None:
    INTER_OP_THREADS = int(INTER_OP_THREADS)

EXECUTION_MODE = os.environ.get("RECOGNIZE_EXECUTION_MODE", "sequential").lower()

# Quantization Preference (attempts to load .quant.onnx if True)
USE_QUANTIZED = os.environ.get("RECOGNIZE_USE_QUANTIZED", "True").lower() in ("true", "1", "yes")

# Known model names (placed in MODELS_DIR by running download_models.py or export_clip.py)
MODEL_NAMES = ["yolov8n", "arcface", "clip_visual"]

# ---------------------------------------------------------------------------
# YOLO settings
# ---------------------------------------------------------------------------
YOLO_INPUT_SIZE = 640
YOLO_CONF_THRESHOLD = 0.25
YOLO_IOU_THRESHOLD = 0.45

# ---------------------------------------------------------------------------
# ArcFace settings
# ---------------------------------------------------------------------------
ARCFACE_INPUT_SIZE = 112

# ---------------------------------------------------------------------------
# CLIP settings
# ---------------------------------------------------------------------------
CLIP_INPUT_SIZE = 224
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
CLIP_TOKENIZER = "openai/clip-vit-base-patch32"

# ---------------------------------------------------------------------------
# File scanner settings
# ---------------------------------------------------------------------------
SCANNER_ENABLED = os.environ.get("RECOGNIZE_SCANNER_ENABLED", "false").lower() in ("true", "1", "yes")
SCANNER_BATCH_SIZE = int(os.environ.get("RECOGNIZE_SCANNER_BATCH_SIZE", "50"))
SCANNER_INTERVAL = int(os.environ.get("RECOGNIZE_SCANNER_INTERVAL", "300"))  # seconds

# ---------------------------------------------------------------------------
# COCO class names for YOLO
# ---------------------------------------------------------------------------
YOLO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]
