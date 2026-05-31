# AI & Vibe Coding Instructions — Recognize AI Backend

Welcome, AI Agent or Vibe Coder! This document outlines the architectural specifications, processing logic, mathematical models, and coding conventions for the `recognize-ai-backend` project. 

Always consult this file before proposing, writing, or refactoring code in this repository.

---

## 🏗️ Architectural Overview & Core Flow

This backend is a lightweight inference server. Its primary job is to take raw image bytes from an HTTP POST request, run them through pre-processing pipelines, execute highly optimized ONNX models, post-process the results, and return standard JSON representations.

### Model Loading Protocol (Lazy Setup)
To guarantee high performance and fast startups, the service employs a **lazy-loading singleton manager** (`ModelManager` in `utils.py`):
1. On application startup, `lifespan()` in `main.py` invokes `model_manager.load_all()`.
2. This attempts to load all models listed under `MODEL_NAMES` in `config.py` into active memory.
3. If a model fails to load (e.g. missing ONNX file, GPU configuration error), **the server does NOT crash**. It stores the error state and continues.
4. When an endpoint is triggered, it calls `model_manager.get(model_name)`. If the model is not loaded, it attempts an **on-the-fly load** before failing.

---

## 🎨 Machine Learning Pipelines

Each model has extremely strict input, output, and normalization requirements. These must match Nextcloud Recognize expectations. **Do not modify preprocessing parameters without explicit user instructions.**

```
                      +-------------------+
                      |   Image Bytes     |
                      +---------+---------+
                                |
          +---------------------+---------------------+
          |                     |                     |
          v                     v                     v
    [ YOLOv8n ]             [ ArcFace ]             [ CLIP ]
  1. Letterbox (640)     1. Face Crop           1. Resize Shortest (224)
  2. Div by 255.0        2. Resize (112x112)    2. Center Crop (224x224)
  3. NCHW Shape          3. Normalise [-1, 1]   3. ImageNet Mean/Std
  4. Run ONNX Session    4. NCHW Shape          4. NCHW Shape
  5. Apply custom NMS    5. L2 Norm Embeddings  5. L2 Norm Embeddings
```

### 1. YOLOv8n (Object Detection)
* **Model ID**: `yolov8n.onnx`
* **Input Shape**: `[1, 3, 640, 640]` (float32)
* **Preprocessing Rules**:
  - **Letterboxing**: Resize the image to fit a `640x640` canvas while *strictly preserving the aspect ratio*. Fill empty margins with gray padding (`114, 114, 114`).
  - **Scale**: Normalize pixel values from `[0, 255]` to `[0.0, 1.0]` by dividing by `255.0`.
  - **Layout**: Transpose to Channel-First format: `[CHW]` and add batch dimension: `[1, C, H, W]`.
* **Output Format**:
  - Transposed output tensor shape: `[8400, 84]` (8400 candidates, 84 features).
  - Columns `0..3` correspond to `[center_x, center_y, width, height]` in the 640x640 space.
  - Columns `4..83` correspond to confidence scores for the 80 COCO classes.
* **Post-processing Rules**:
  - Filter out detections where class confidence is below `YOLO_CONF_THRESHOLD` (default `0.25`).
  - Convert `[center_x, center_y, width, height]` back to original coordinates: `[x_min, y_min, x_max, y_max]`, subtracting the `pad_w` and `pad_h` and dividing by the scale `ratio`.
  - Run Non-Maximum Suppression (NMS) with an IOU threshold of `YOLO_IOU_THRESHOLD` (default `0.45`) to deduplicate overlaps.

### 2. ArcFace (Face Embeddings)
* **Model ID**: `arcface.onnx`
* **Input Shape**: `[1, 3, 112, 112]` (float32)
* **Preprocessing Rules**:
  - Crop face region using YOLO object detections (filtered for classes identified as `"person"`). If no person is detected, fall back to crop the full image bounds.
  - Resize cropped face directly to `112x112` using bilinear interpolation.
  - **Normalization**: Map pixel values from `[0, 255]` to a range of `[-1.0, 1.0]` using `(pixel - 127.5) / 128.0`.
  - Transpose to Channel-First format: `[CHW]` and add batch dimension: `[1, C, H, W]`.
* **Post-processing Rules**:
  - Flatten the final embeddings tensor.
  - **L2-Normalize** the embedding vector before returning to ensure cosine similarity calculation consistency.

### 3. CLIP ViT-B/32 (Semantic Search)
* **Model ID**: `clip_visual.onnx`
* **Input Shape**: `[1, 3, 224, 224]` (float32)
* **Preprocessing Rules**:
  - Resize the shortest edge of the image to `224` pixels while preserving the aspect ratio.
  - Apply a center crop of exactly `224x224` pixels.
  - Scale pixel values to `[0.0, 1.0]` by dividing by `255.0`.
  - **ImageNet Normalization**: Normalize channels with:
    - `mean = [0.48145466, 0.4578275, 0.40821073]`
    - `std  = [0.26862954, 0.26130258, 0.27577711]`
    - Formula: `(pixel - mean) / std`
  - Transpose to Channel-First format: `[CHW]` and add batch dimension: `[1, C, H, W]`.
* **Execution Input Variants**:
  - Some CLIP models expect `pixel_values` or `input`. Others expect auxiliary inputs like `input_ids` and `attention_mask`. The code dynamically fills these to support different ONNX compiler targets.
* **Post-processing Rules**:
  - Extract the visual embeddings, flatten them, and apply **L2-Normalization**.

---

## 🛠️ Vibe Coding & Agent Guidelines

When modifying this repository, follow these rules to ensure the codebase remains clean, fast, and easy to maintain.

### 1. Maintain Code Modularity & Consistency
- **Do not introduce heavy framework dependencies** (e.g. PyTorch, torchvision, transformers) to `inference.py` or `main.py`. This project is built specifically to use **Pillow**, **numpy**, and **onnxruntime** to stay extremely lightweight and fast to containerize.
- Maintain the lazy-loading flow in `ModelManager` if you add new models.
- All model coordinates must reside in `config.py` rather than being hardcoded in processing methods.

### 2. Standard Coding Conventions
- **Language**: Python 3.10+
- **Style**: PEP-8 compliant. Focus on explicit typings (`typing.Optional`, `list[dict]`, `np.ndarray`, etc.).
- **Async Execution**: Use async context only for API router endpoints (`FastAPI` UploadFile processing) and keep inference functions synchronous as they are compute-bound CPU/GPU operations. Running synchronous heavy inference on the event loop is okay because uvicorn routes can be executed in worker threads if wrapped, or standard CPU bounds can be mitigated by scaling workers.

### 3. Adding a New Model Endpoint
If you are instructed to add a new model pipeline:
1. Save the model in ONNX format.
2. Put the default configurations (input dimensions, confidence thresholds, model name) in `config.py`.
3. Add the model name to `MODEL_NAMES` so `ModelManager` loads it and tracks its status.
4. Add a preprocessing, session running, and postprocessing method in `inference.py`.
5. Add a FastAPI route in `main.py` that uploads an image, invokes the inference method, and returns structured JSON responses.
6. Add corresponding test coverage to `test_client.py`.

---

## 🚨 Troubleshooting Hand-off

### 1. ONNX Runtime execution provider issues
* **Symptom**: `Failed to load model ...: [ONNXRuntimeError] : 1 : FAIL : Load model failed...` or CUDA provider initialization warnings.
* **Solution**: Check the list of available providers using `ort.get_available_providers()`. On standard machines without an Nvidia GPU, ensure `CPUExecutionProvider` is listed first or is the fallback. If `onnxruntime-gpu` is installed, verify the host has the matching CUDA and cuDNN DLLs (e.g. `cudart64_110.dll` or `cudart64_12.dll`).

### 2. PIL Image File Crashes
* **Symptom**: `UnidentifiedImageError: cannot identify image file`.
* **Solution**: Ensure input raw bytes are wrapped in a `BytesIO` buffer before attempting to load via `Image.open`. Verify the image type is transformed with `.convert("RGB")` to strip alpha channels (PNG) or CMYK configurations which crash numpy array casting operations.

---

## 🛡️ Error Handling Patterns

The project uses a two-layer error handling strategy: **graceful degradation** at the inference layer and **proper HTTP responses** at the endpoint layer.

### Inference Layer (`inference.py`)
All `process_*` functions (`process_yolo`, `process_arcface`, `process_clip`) follow the same pattern:
1. Attempt to acquire the ONNX session via `model_manager.get(model_name)`.
2. If the session is `None`, attempt an on-the-fly load via `model_manager.load(model_name)`.
3. If the load also fails, **return an empty list `[]`** — never raise an exception for model-not-loaded.

This means callers can always expect a `list` return type. An empty list signals "no results" without crashing the server. **Agents must NOT change this convention.** Do not convert these returns to exceptions or `None`.

### Endpoint Layer (`main.py`)
The FastAPI endpoints wrap inference calls and translate failures into appropriate HTTP status codes:

| HTTP Code | Condition | Example |
|-----------|-----------|---------|
| **200** | Successful inference | Normal detection/embedding response |
| **413** | Upload exceeds `MAX_UPLOAD_SIZE` | File larger than 20 MB |
| **422** | Bad input (malformed upload, missing file) | FastAPI's automatic validation |
| **500** | Unexpected inference crash | Numpy error, corrupted model output |
| **503** | Model unavailable (not loaded, failed to load) | ONNX file missing or GPU mismatch |

**Agents must NOT change this error hierarchy.** The 503 vs 500 distinction is intentional: 503 tells Nextcloud to retry later (the model may load on a subsequent attempt), while 500 indicates a bug that needs developer attention.

---

## 🔐 Security Considerations

### Upload Validation
- Uploads are validated against `MAX_UPLOAD_SIZE` (defined in `config.py`, default **20 MB**). Requests exceeding this limit are rejected with HTTP 413 before any processing occurs.
- This prevents denial-of-service via oversized image uploads consuming memory during numpy array construction.

### Image Safety
- All uploaded images are immediately converted to RGB via `Image.open(BytesIO(data)).convert("RGB")` (see `_load_image()` in `inference.py`).
- This conversion strips alpha channels (PNG), CMYK color spaces, and palette-mode images that could cause numpy casting failures or produce unexpected tensor shapes.
- Agents must preserve this `.convert("RGB")` call. Removing it will cause crashes on PNG and CMYK inputs.

### Authentication
- **There is currently no authentication on the API endpoints.** Nextcloud handles authentication and authorization at the reverse proxy layer before requests reach this service.
- Agents must NOT add authentication middleware unless explicitly instructed. Adding auth here would break the Nextcloud ExApp integration contract.

---

## 📝 Logging Conventions

The project uses Python's built-in `logging` module consistently across all files.

### Setup
- Logger instances are created with `logging.getLogger(__name__)` at module level in every file.
- The root log format is configured in `main.py`:
  ```python
  logging.basicConfig(
      level=logging.INFO,
      format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
  )
  ```

### Usage Rules
| Level | When to Use | Example |
|-------|-------------|---------|
| `logger.info()` | Normal operational flow | `"YOLO detected 5 objects"`, `"Model 'arcface' loaded successfully."` |
| `logger.warning()` | Recoverable issues that do not crash the request | `"CLIP visual model not loaded, attempting on-the-fly load …"` |
| `logger.exception()` | Inside `except` blocks — auto-attaches the traceback | Inference crash, unexpected numpy error |
| `logger.error()` | Errors outside of except blocks | Configuration validation failure |

### Hard Rule
**Agents must NOT use `print()` for diagnostics.** All output must go through `logger.*()` calls so that log levels, timestamps, and module names are consistent. The only acceptable use of `print()` is in standalone scripts like `download_models.py` that run outside the server context.

---

## ⚡ Concurrency Model

### Why Endpoints Are `def` (Not `async def`)
The analysis endpoints (`/analyze/objects`, `/analyze/faces`, `/analyze/semantic`) should use plain `def` (not `async def`) so that FastAPI automatically runs them in a **thread pool** (`anyio` worker threads). This is critical because:

1. **ONNX Runtime inference is CPU-bound** — `session.run()` can take 50–500ms depending on model size and hardware.
2. If endpoints were `async def`, the inference call would **block the event loop**, preventing the server from handling concurrent requests (health checks, other uploads) during that time.
3. With `def` endpoints, FastAPI offloads each request to a thread, keeping the event loop free.

### Rules for Agents
- **Analysis endpoints** (`/analyze/*`): Must be `def`, not `async def`. If you change them to `async def`, you **must** also wrap the inference call in `await asyncio.get_event_loop().run_in_executor(None, ...)` — otherwise the server will serialize all requests.
- **Lifespan context**: Uses `async def` because it runs during startup/shutdown, not during request handling.
- **Lightweight GET endpoints** (`/health`, `/models/status`): May be `async def` since they return immediately without compute-heavy work.

### Thread Safety
The `ModelManager` singleton is shared across threads. ONNX Runtime `InferenceSession.run()` is thread-safe by design (it holds an internal mutex). However, if you add mutable state to `ModelManager`, you must add proper locking (e.g. `threading.Lock`).

---

## 🔌 CLIP Input Name Fallback Logic

The `process_clip` function in `inference.py` contains a dynamic input name resolution system. This exists because **different CLIP export methods produce ONNX models with different input node names**, and the code must handle all variants without modification.

### How It Works
```python
inputs = {}
for i in session.get_inputs():
    if i.name in ["pixel_values", "input"]:
        inputs[i.name] = tensor            # Image tensor
    elif i.name == "input_ids":
        inputs[i.name] = np.array([[49406, 49407]], dtype=np.int64)   # SOT + EOT tokens
    elif i.name == "attention_mask":
        inputs[i.name] = np.array([[1, 1]], dtype=np.int64)           # Both tokens attended

# Fallback: if no known name matched, use the first input node
if not inputs and session.get_inputs():
    inputs[session.get_inputs()[0].name] = tensor
```

### Input Name Variants

| Input Name | Source | What It Gets |
|------------|--------|-------------|
| `pixel_values` | HuggingFace `CLIPModel` export | The preprocessed image tensor |
| `input` | OpenAI CLIP export / generic ONNX export | The preprocessed image tensor |
| `input_ids` | HuggingFace export (text encoder inputs) | Dummy SOT/EOT token pair `[49406, 49407]` |
| `attention_mask` | HuggingFace export (text encoder inputs) | Ones `[1, 1]` for both tokens |
| *(first input)* | Unknown/custom exports | Fallback — assumes it wants the image tensor |

### Why This Matters
- The dummy `input_ids` and `attention_mask` values are the CLIP start-of-text (SOT = 49406) and end-of-text (EOT = 49407) tokens. They are required by HuggingFace-exported models that fuse the visual and text encoders into a single ONNX graph.
- **Agents must preserve this fallback logic.** Hardcoding a single input name (e.g. always using `"pixel_values"`) will break compatibility with alternative CLIP exports.
- If adding support for a new CLIP variant, add its input name to the existing `if/elif` chain — do not replace the current logic.
