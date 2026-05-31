"""
Nextcloud Recognize ExApp — FastAPI entrypoint.

Supports dual-mode operation:
  - Standalone: python main.py (dev mode, no auth, root paths)
  - ExApp: Managed by Nextcloud AppAPI (auth middleware, /api/v1 prefix)
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from config import (
    HOST, PORT, MAX_UPLOAD_SIZE, EXAPP_MODE, APP_VERSION,
    YOLO_CONF_THRESHOLD, YOLO_IOU_THRESHOLD, USE_QUANTIZED,
    SCANNER_ENABLED,
)
from utils import model_manager
from inference import process_yolo, process_arcface, process_clip, process_clip_text
from scanner import file_scanner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan – download & load models on startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting model loading …")
    model_manager.load_all()
    status = model_manager.status()
    loaded = sum(1 for v in status.values() if v["loaded"])
    logger.info(f"Model loading complete: {loaded}/{len(status)} models ready.")

    # Setup ExApp lifecycle handlers (no-op in standalone mode)
    from nc_app import setup_exapp
    setup_exapp(app)

    # Start file scanner if enabled
    if SCANNER_ENABLED and EXAPP_MODE:
        file_scanner.start()

    yield

    # Shutdown
    file_scanner.stop()
    logger.info("Shutting down Recognize ExApp.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Nextcloud Recognize AI",
    description="Python-based AI microservice for Nextcloud Recognize using ONNX Runtime.",
    version=APP_VERSION,
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_and_validate_upload(file: UploadFile) -> bytes:
    """Read upload bytes and validate size and basic image sanity."""
    image_bytes = file.file.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=422, detail="Uploaded file is empty.")
    if len(image_bytes) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(image_bytes)} bytes). Maximum is {MAX_UPLOAD_SIZE} bytes.",
        )
    return image_bytes


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class TextQuery(BaseModel):
    text: str


class ScannerConfig(BaseModel):
    enabled: bool
    batch_size: int = None
    interval: int = None


# ---------------------------------------------------------------------------
# API Router (versioned — used in both modes)
# ---------------------------------------------------------------------------

router = APIRouter()


@router.get("/health")
async def health_check():
    """Quick liveness probe."""
    return {"status": "ok", "message": "Recognize AI is running.", "version": APP_VERSION}


@router.get("/models/status")
async def models_status():
    """Return which models are loaded (or why they failed)."""
    return model_manager.status()


@router.post("/analyze/objects")
def analyze_objects(file: UploadFile = File(...)):
    """Detect objects in the uploaded image using YOLOv8."""
    image_bytes = _read_and_validate_upload(file)
    try:
        results = process_yolo(image_bytes)
    except Exception as e:
        logger.exception("YOLO inference failed")
        raise HTTPException(status_code=500, detail=f"Object detection failed: {e}")
    if results is None:
        raise HTTPException(status_code=503, detail="YOLO model is not available.")
    return JSONResponse(content=results)


@router.post("/analyze/faces")
def analyze_faces(file: UploadFile = File(...)):
    """Detect faces and compute ArcFace embeddings."""
    image_bytes = _read_and_validate_upload(file)
    try:
        results = process_arcface(image_bytes)
    except Exception as e:
        logger.exception("ArcFace inference failed")
        raise HTTPException(status_code=500, detail=f"Face recognition failed: {e}")
    if results is None:
        raise HTTPException(status_code=503, detail="ArcFace model is not available.")
    return JSONResponse(content=results)


@router.post("/analyze/semantic")
def analyze_semantic(file: UploadFile = File(...)):
    """Compute a CLIP image embedding for semantic search."""
    image_bytes = _read_and_validate_upload(file)
    try:
        results = process_clip(image_bytes)
    except Exception as e:
        logger.exception("CLIP inference failed")
        raise HTTPException(status_code=500, detail=f"Semantic embedding failed: {e}")
    if results is None:
        raise HTTPException(status_code=503, detail="CLIP model is not available.")
    return JSONResponse(content=results)


@router.post("/analyze/text")
def analyze_text(query: TextQuery):
    """Compute a CLIP text embedding for a natural language search query."""
    if not query.text.strip():
        raise HTTPException(status_code=422, detail="Text query cannot be empty.")
    try:
        results = process_clip_text(query.text)
    except Exception as e:
        logger.exception("CLIP text embedding failed")
        raise HTTPException(status_code=500, detail=f"Text embedding failed: {e}")
    if not results:
        raise HTTPException(status_code=503, detail="CLIP model is not available.")
    return JSONResponse(content=results)


# ---------------------------------------------------------------------------
# Admin API endpoints (consumed by the Vue.js frontend)
# ---------------------------------------------------------------------------

@router.get("/config")
async def get_config():
    """Return current configuration values (non-sensitive) for the admin panel."""
    return {
        "version": APP_VERSION,
        "exapp_mode": EXAPP_MODE,
        "max_upload_size": MAX_UPLOAD_SIZE,
        "yolo_conf_threshold": YOLO_CONF_THRESHOLD,
        "yolo_iou_threshold": YOLO_IOU_THRESHOLD,
        "use_quantized": USE_QUANTIZED,
    }


@router.get("/gpu-info")
async def get_gpu_info():
    """Return detected GPU/execution provider information."""
    return model_manager.get_gpu_info()


@router.get("/scanner/status")
async def scanner_status():
    """Return current file scanner status and stats."""
    return file_scanner.status()


@router.post("/scanner/toggle")
async def scanner_toggle(config: ScannerConfig):
    """Enable/disable the file scanner with optional configuration."""
    file_scanner.configure(
        enabled=config.enabled,
        batch_size=config.batch_size,
        interval=config.interval,
    )
    return file_scanner.status()


# ---------------------------------------------------------------------------
# Mount routes
# ---------------------------------------------------------------------------

# In ExApp mode, all routes are under /api/v1
# In standalone mode, routes are also mounted at root for backward compat
if EXAPP_MODE:
    app.include_router(router, prefix="/api/v1")
else:
    # Standalone dev mode — mount at root AND /api/v1
    app.include_router(router)
    app.include_router(router, prefix="/api/v1")


# ---------------------------------------------------------------------------
# Direct execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from nc_app import run_exapp

    # Try ExApp mode first; fall back to standalone uvicorn
    if not run_exapp("main:app"):
        import uvicorn
        uvicorn.run("main:app", host=HOST, port=PORT, reload=True)
