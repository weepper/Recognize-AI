# Gemini Instructions — Recognize AI Backend

This is a FastAPI + ONNX Runtime AI microservice for Nextcloud Recognize.

## Architecture
- **API**: FastAPI (main.py) with sync analysis endpoints for automatic threadpooling
- **Inference**: Pure numpy + Pillow + onnxruntime pipelines (inference.py)
- **Models**: YOLOv8n (objects), ArcFace (face embeddings), CLIP ViT-B/32 (semantic embeddings)
- **Management**: Lazy-loading ModelManager singleton (utils.py)
- **Config**: Environment-driven constants (config.py)

## Key Conventions
- Read `ai_instructions.md` for detailed preprocessing math and agent guidelines
- All model parameters (input sizes, normalization constants, thresholds) live in `config.py`
- No heavy ML frameworks (PyTorch, TF, transformers) in runtime code
- Use `logging.getLogger(__name__)` — never `print()`
- Analysis endpoints are `def` (not `async def`) for FastAPI threadpool execution
- New models: add to `config.MODEL_NAMES` → implement in `inference.py` → route in `main.py`
- Error pattern: inference returns `[]` on failure; endpoints return HTTP 503/500/422/413

## Testing
- Start server: `python main.py`
- Run tests: `python test_client.py`
- Models: `python download_models.py` (downloads/exports all three models)

## ExApp Integration (Nextcloud)
- **Dual-mode**: Runs standalone (`python main.py`) or as a Nextcloud ExApp managed by AppAPI
- **nc-py-api lifecycle**: `nc_app.py` registers routes with Nextcloud via `nc-py-api`; active only when `APP_ID`/`APP_SECRET` env vars are present
- **Frontend**: Vue.js admin settings panel in `src/`, built with webpack
- **Scanner**: `scanner.py` provides background file processing (batch scanning of Nextcloud media)
- **Manifest**: `appinfo/info.xml` declares the ExApp ID, version, deploy config, and admin settings
- **CI/CD**: `.github/workflows/build-docker.yml` builds multi-platform Docker images → `ghcr.io/pener/recognize-ai-backend`

