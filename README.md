# Nextcloud Recognize ExApp (AI Backend)

[![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0-009688.svg?style=flat&logo=FastAPI&logoColor=white)](https://fastapi.tiangolo.com/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-1.17.0-005C99.svg?style=flat&logo=onnx&logoColor=white)](https://onnxruntime.ai/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)

An extremely fast, lightweight, and hardware-accelerated **AI microservice** built specifically as an **External App (ExApp)** for **Nextcloud Recognize**. It serves object detection, facial recognition/embeddings, and semantic image embeddings using **ONNX Runtime** with automatic execution provider selection (e.g., CUDA for Nvidia GPUs, or highly optimized CPU threads).

```
                      +-----------------------------+
                      | Nextcloud Recognize (App)   |
                      +--------------+--------------+
                                     |
                       REST API /    |  1. Upload Image
                       HTTP POST     |  2. Get JSON Results
                                     v
                 +-------------------+--------------------+
                 | FastAPI ExApp Backend (Port 8000)      |
                 +-------------------+--------------------+
                                     |
            +------------------------+------------------------+
            |                        |                        |
            v                        v                        v
  +------------------+     +------------------+     +------------------+
  |    YOLOv8n       |     |    ArcFace       |     |     CLIP         |
  |  (Object Det.)   |     | (Face Embeddings)|     | (Semantic Search)|
  +--------+---------+     +--------+---------+     +--------+---------+
           |                        |                        |
           +------------------------+------------------------+
                                     |
                                     v
                       +-------------+-------------+
                       | ONNX Runtime Engine       |
                       | - CUDA (GPU Accelerated)  |
                       | - CPU (Fallback / Native) |
                       +---------------------------+
```

---

## Key Features

- 🏎️ **Supercharged Inference**: Leverages **ONNX Runtime** which compiles neural networks into highly-optimized graph representations, running orders of magnitude faster than standard PyTorch/TensorFlow.
- 🧠 **Smart Lazy-Loading**: Keeps the API startup light. Models are loaded on startup and tracked by a singleton `ModelManager`, with on-the-fly fallback loading if needed.
- 🔌 **Seamless Integration**: Fully compatible with Nextcloud Recognize's external API specifications.
- 🛡️ **Robust Error Handling**: Upload validation (size limits, empty file detection), proper HTTP error codes (503/500/422/413), and graceful degradation when models are unavailable.
- 🎨 **Unified Pipeline**: Implements three state-of-the-art vision pipelines:
  - **Object Detection**: YOLOv8n (80 COCO classes) with custom letterboxing and Non-Maximum Suppression (NMS).
  - **Facial Recognition**: ArcFace embeddings (512-d). Auto-detects human faces using YOLO bounding boxes and extracts normalised facial embeddings.
  - **Semantic Search**: CLIP (ViT-B/32) image encoder generating high-fidelity vector representations for natural language search.

---

## Project Structure

```
recognize-ai-backend/
├── config.py                # Global settings, class lists, thresholds, env vars
├── main.py                  # FastAPI server entrypoint and endpoint routes
├── inference.py             # Image preprocessing, NMS, and ONNX execution pipelines
├── utils.py                 # Lazy ModelManager and ONNX session initialization
├── nc_app.py                # Nextcloud ExApp lifecycle (nc-py-api registration)
├── scanner.py               # Background file scanner for automatic media classification
├── download_models.py       # Unified downloader/exporter for YOLO, ArcFace, and CLIP
├── export_clip.py           # Standalone CLIP exporter (legacy, use download_models.py)
├── test_client.py           # Functional testing client (sends test requests)
├── models/                  # ONNX model storage (.onnx files, gitignored)
├── appinfo/
│   └── info.xml             # Nextcloud ExApp manifest (app ID, version, deploy config)
├── src/
│   └── main.js              # Vue.js admin settings panel (webpack-built frontend)
├── requirements.txt         # Runtime dependencies
├── requirements-dev.txt     # Development & model export dependencies
├── Dockerfile               # Multi-stage CPU/GPU production container
├── .env.example             # Environment variable reference
├── .gitignore               # Git ignore rules
├── ai_instructions.md       # Detailed AI agent & vibe coding guidelines
├── GEMINI.md                # Gemini CLI / Antigravity agent rules
├── .cursorrules             # Cursor AI agent rules
├── .clinerules              # Cline / Roo-Code agent rules
└── .github/
    ├── copilot-instructions.md  # GitHub Copilot agent rules
    └── workflows/
        └── build-docker.yml     # CI/CD: build & push Docker image to ghcr.io
```

---

## ⚡ Quick Start

### 1. Prerequisite: Environment Setup
We recommend using a virtual environment (Python 3.10+):

```bash
# Clone the repository and navigate inside
cd recognize-ai-backend

# Create and activate a virtual environment
python -m venv .venv
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Linux / macOS:
source .venv/bin/activate

# Install core runtime dependencies
pip install -r requirements.txt

# Install development & model export dependencies
pip install -r requirements-dev.txt
```

### 2. Download and Export Models
Download and export all three `.onnx` models into the `models/` directory with a single command:

```bash
python download_models.py
```

Or download models selectively:
```bash
python download_models.py yolo      # YOLOv8n only (~12 MB)
python download_models.py arcface   # ArcFace only (~260 MB)
python download_models.py clip      # CLIP ViT-B/32 only (~605 MB)
```

### 3. Launch the Server
Start the FastAPI server:

```bash
python main.py
```
The server will start on [http://127.0.0.1:8000](http://127.0.0.1:8000). The startup lifespan will attempt to pre-load all configured ONNX models.

### 4. Docker (Alternative)
Build and run with Docker:
```bash
# CPU build
docker build -t recognize-backend .
docker run -p 8000:8000 -v ./models:/app/models recognize-backend

# GPU build (requires NVIDIA Container Toolkit)
docker build --build-arg GPU=true -t recognize-backend-gpu .
docker run --gpus all -p 8000:8000 -v ./models:/app/models recognize-backend-gpu
```

---

## 🔌 Nextcloud ExApp Installation

This backend can run as a **Nextcloud External App (ExApp)** via the [AppAPI](https://github.com/cloud-py-api/app_api) framework, giving Nextcloud full lifecycle control over the container.

### Prerequisites
- **Nextcloud 30+** with the **AppAPI** app installed and enabled
- A configured **Deploy Daemon** (Docker Socket Proxy or similar)
- Docker running on the Nextcloud host or a remote Docker daemon

### Step 1: Setup Deploy Daemon
In your Nextcloud Admin settings, navigate to **AppAPI → Deploy Daemons** and configure a Docker-based daemon. This tells AppAPI how to pull and manage ExApp containers.

### Step 2: Install the ExApp
Register and deploy the ExApp using the Nextcloud `occ` CLI:

```bash
sudo -u www-data php occ app_api:app:register recognize_ai \
    --info-xml https://raw.githubusercontent.com/pener/recognize-ai-backend/main/appinfo/info.xml \
    --json-info "{\"appid\":\"recognize_ai\",\"name\":\"Recognize AI\",\"daemon_config_name\":\"docker_install\",\"version\":\"1.0.0\",\"secret\":\"auto\",\"port\":8000,\"routes\":[{\"url\":\".*\",\"verb\":\"GET,POST,PUT,DELETE\",\"access_level\":\"ADMIN\",\"headers_to_exclude\":[]}]}" \
    --force-scopes \
    --wait-finish
```

AppAPI will automatically:
- Pull the image from `ghcr.io/pener/recognize-ai-backend:latest`
- Create and start the container
- Inject `APP_ID`, `APP_SECRET`, and `NEXTCLOUD_URL` environment variables

### Step 3: Configure in Admin Settings
After installation, navigate to **Admin Settings → Recognize AI** to:
- View **model loading status** (YOLOv8n, ArcFace, CLIP)
- Check **GPU information** and execution provider
- Enable and configure the **background file scanner**

### Standalone vs. ExApp Mode
| Feature | Standalone | ExApp |
|---|---|---|
| Startup | `python main.py` | Managed by AppAPI |
| Auth | None (open API) | AppAPI shared secret |
| Scanner | Not available | Background file scanning |
| Admin Panel | Not available | Vue.js settings UI |
| Config | `.env` file | Nextcloud Admin Settings |

> **Note**: When developing locally, use standalone mode (`python main.py`). The ExApp lifecycle hooks in `nc_app.py` are only active when `APP_ID` and `APP_SECRET` environment variables are present.

### Admin Panel Features
The ExApp includes a Vue.js admin settings panel (`src/`) providing:
- **Model Status Dashboard**: Real-time loading state and health of all three AI models
- **GPU Information**: Active ONNX execution providers, CUDA availability, and device details
- **File Scanner Controls**: Enable/disable the background scanner, set batch size, and scan interval

---

## 🔍 API Documentation

### 1. Liveness Check
Checks if the backend microservice is alive.
* **URL**: `GET /health`
* **Response**:
  ```json
  { "status": "ok", "message": "Recognize ExApp is running." }
  ```

### 2. Model Loading Status
Returns the active state and loading status of all backend models.
* **URL**: `GET /models/status`
* **Response**:
  ```json
  {
    "yolov8n": { "loaded": true, "error": null },
    "arcface": { "loaded": true, "error": null },
    "clip_visual": { "loaded": true, "error": null }
  }
  ```

### 3. Object Detection (YOLOv8n)
Detects 80 classes of objects within an image.
* **URL**: `POST /analyze/objects`
* **Payload**: Multipart form-data with key `file` (image bytes).
* **Response** (`200`):
  ```json
  [
    { "class": "person", "score": 0.8942, "box": [120, 45, 340, 580] },
    { "class": "tie", "score": 0.7612, "box": [210, 150, 245, 310] }
  ]
  ```
  *(Bounding boxes are `[x_min, y_min, x_max, y_max]` in original image pixel coordinates).*

### 4. Facial Recognition (ArcFace)
Crops faces using person detections and extracts 512-dimensional L2-normalized face embeddings.
* **URL**: `POST /analyze/faces`
* **Payload**: Multipart form-data with key `file` (image bytes).
* **Response** (`200`):
  ```json
  [
    { "embedding": [0.0241, -0.0152, "... 512 values ..."], "box": [120, 45, 340, 580] }
  ]
  ```

### 5. Semantic Search (CLIP)
Generates high-fidelity visual embeddings for semantic cataloging and text-to-image queries.
* **URL**: `POST /analyze/semantic`
* **Payload**: Multipart form-data with key `file` (image bytes).
* **Response** (`200`):
  ```json
  [ { "embedding": [-0.0118, 0.0345, "... 512 values ..."] } ]
  ```

### Error Responses

| HTTP Code | Condition |
|-----------|-----------|
| `413` | File exceeds `MAX_UPLOAD_SIZE` (default 20 MB) |
| `422` | Bad input (empty file, missing field) |
| `500` | Unexpected inference crash |
| `503` | Model unavailable (not loaded) |

---

## 🧪 Functional Testing

Verify the entire setup using the built-in `test_client.py` script. It tests all 5 endpoints (`/health`, `/models/status`, and all three analysis routes):

```bash
python test_client.py
```

---

## ⚙️ Configuration
Customize the runtime via environment variables (see `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `RECOGNIZE_MODELS_DIR` | `./models` | Directory storing `.onnx` models |
| `RECOGNIZE_HOST` | `127.0.0.1` | Server host address |
| `RECOGNIZE_PORT` | `8000` | Server port |
| `RECOGNIZE_MAX_UPLOAD_SIZE` | `20971520` (20 MB) | Maximum upload file size in bytes |
| `RECOGNIZE_ONNX_PROVIDERS` | `CUDAExecutionProvider,CPUExecutionProvider` | Ordered list of ONNX providers |

To run on GPU, ensure you have the appropriate CUDA Toolkit installed along with `onnxruntime-gpu`.

---

## 🤖 Contributing (AI Agents & Vibe Coders)

Before making changes to this codebase, **read [`ai_instructions.md`](ai_instructions.md)**. It contains:
- Exact preprocessing math for each model (input shapes, normalization constants, tensor layouts)
- Error handling hierarchy and concurrency model
- Step-by-step guide for adding new model endpoints
- Security constraints and logging conventions

Agent-specific rule files are also available:
- **Gemini**: [`GEMINI.md`](GEMINI.md)
- **Cursor**: [`.cursorrules`](.cursorrules)
- **Cline/Roo-Code**: [`.clinerules`](.clinerules)
- **GitHub Copilot**: [`.github/copilot-instructions.md`](.github/copilot-instructions.md)
