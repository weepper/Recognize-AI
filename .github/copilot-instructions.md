# GitHub Copilot Instructions — Recognize AI Backend

This is a FastAPI + ONNX Runtime inference microservice for Nextcloud Recognize.

## Project Conventions
- **Read `ai_instructions.md`** in the project root for full architecture documentation
- Python 3.10+ with explicit type hints
- PEP-8 compliant, `logging.getLogger(__name__)` for all diagnostics (no `print()`)
- Runtime dependencies limited to: FastAPI, uvicorn, Pillow, numpy, onnxruntime, httpx

## Architecture Rules
- All model constants (input sizes, normalization mean/std, thresholds) must be in `config.py`
- All models are loaded via `ModelManager` singleton in `utils.py` using lazy-loading
- Analysis endpoints (`/analyze/*`) in `main.py` must be plain `def` (not `async def`) so FastAPI runs them in a thread pool, preventing event loop blocking during ONNX inference
- Inference functions in `inference.py` return `[]` on model-not-loaded (graceful degradation)
- Endpoints wrap inference calls in try/except and return proper HTTP codes (503, 500, 422, 413)

## Adding a New Model
1. Export to ONNX format, add download logic to `download_models.py`
2. Add constants (input size, normalization params) to `config.py`
3. Register model name in `config.MODEL_NAMES`
4. Implement preprocessing, inference, and postprocessing in `inference.py`
5. Add a `def` POST route in `main.py`
6. Add test coverage in `test_client.py`

## ExApp Conventions
- **Dual-mode operation**: Standalone (`python main.py`) or Nextcloud ExApp. ExApp mode activates when `APP_ID` and `APP_SECRET` env vars are present (injected by AppAPI).
- **nc-py-api lifecycle**: `nc_app.py` handles ExApp registration with Nextcloud via `nc-py-api`. Keep all nc-py-api imports isolated to this file.
- **Vue.js frontend**: Admin settings panel in `src/`, built with webpack. Follow Nextcloud Vue component conventions.
- **Scanner**: `scanner.py` handles background file processing — fetches unscanned files from Nextcloud in configurable batches.
- **Manifest**: `appinfo/info.xml` is the ExApp manifest. Keep `<version>` in sync with git tags and Docker image tags.

