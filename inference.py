"""
Real ONNX inference for YOLO object detection, ArcFace face recognition,
and CLIP semantic embeddings.
"""
import logging
from io import BytesIO
from typing import Optional

import numpy as np
from PIL import Image

from config import (
    YOLO_INPUT_SIZE,
    YOLO_CONF_THRESHOLD,
    YOLO_IOU_THRESHOLD,
    YOLO_CLASSES,
    ARCFACE_INPUT_SIZE,
    CLIP_INPUT_SIZE,
    CLIP_MEAN,
    CLIP_STD,
)
from utils import model_manager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_image(image_bytes: bytes) -> Image.Image:
    """Open raw bytes as a PIL RGB image."""
    return Image.open(BytesIO(image_bytes)).convert("RGB")


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """Simple non-maximum suppression. Returns indices to keep."""
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        mask = iou <= iou_threshold
        order = order[1:][mask]

    return keep


# ---------------------------------------------------------------------------
# YOLO v8  (object detection)
# ---------------------------------------------------------------------------

def _preprocess_yolo(image: Image.Image) -> tuple[np.ndarray, tuple[int, int], float, tuple[float, float]]:
    """
    Letterbox-resize to YOLO_INPUT_SIZE x YOLO_INPUT_SIZE, normalise to [0,1],
    and return (input_tensor, original_size, ratio, (pad_w, pad_h)).
    """
    orig_w, orig_h = image.size
    ratio = min(YOLO_INPUT_SIZE / orig_w, YOLO_INPUT_SIZE / orig_h)
    new_w, new_h = int(orig_w * ratio), int(orig_h * ratio)
    pad_w = (YOLO_INPUT_SIZE - new_w) / 2.0
    pad_h = (YOLO_INPUT_SIZE - new_h) / 2.0

    resized = image.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (YOLO_INPUT_SIZE, YOLO_INPUT_SIZE), (114, 114, 114))
    canvas.paste(resized, (int(pad_w), int(pad_h)))

    arr = np.asarray(canvas, dtype=np.float32) / 255.0  # HWC, [0,1]
    arr = arr.transpose(2, 0, 1)  # CHW
    arr = np.expand_dims(arr, 0)  # NCHW
    return arr, (orig_w, orig_h), ratio, (pad_w, pad_h)


def process_yolo(image_bytes: bytes) -> list[dict]:
    """Run YOLOv8n object detection.  Returns list of {class, score, box}."""
    session = model_manager.get("yolov8n")
    if session is None:
        logger.warning("YOLO model not loaded, attempting on-the-fly load …")
        if not model_manager.load("yolov8n"):
            return []
        session = model_manager.get("yolov8n")
        if session is None:
            return []

    image = _load_image(image_bytes)
    input_tensor, (orig_w, orig_h), ratio, (pad_w, pad_h) = _preprocess_yolo(image)

    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_tensor})

    # YOLOv8 output shape: (1, 84, 8400) → transpose to (8400, 84)
    preds = outputs[0]
    if preds.ndim == 3:
        preds = preds[0]
    if preds.shape[0] < preds.shape[1]:
        preds = preds.T  # now (8400, 84)

    # Columns: cx, cy, w, h, class_scores×80
    cx, cy, w, h = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
    class_scores = preds[:, 4:]
    class_ids = class_scores.argmax(axis=1)
    confidences = class_scores[np.arange(len(class_ids)), class_ids]

    # Filter by confidence
    mask = confidences >= YOLO_CONF_THRESHOLD
    cx, cy, w, h = cx[mask], cy[mask], w[mask], h[mask]
    class_ids = class_ids[mask]
    confidences = confidences[mask]

    # Convert centre-xywh → xyxy in original image coordinates
    x1 = (cx - w / 2 - pad_w) / ratio
    y1 = (cy - h / 2 - pad_h) / ratio
    x2 = (cx + w / 2 - pad_w) / ratio
    y2 = (cy + h / 2 - pad_h) / ratio
    boxes = np.stack([x1, y1, x2, y2], axis=1)

    # Clip to image bounds
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)

    # NMS
    keep = _nms(boxes, confidences, YOLO_IOU_THRESHOLD)

    results = []
    for idx in keep:
        cid = int(class_ids[idx])
        label = YOLO_CLASSES[cid] if cid < len(YOLO_CLASSES) else f"class_{cid}"
        results.append({
            "class": label,
            "score": round(float(confidences[idx]), 4),
            "box": [round(float(v)) for v in boxes[idx]],
        })

    logger.info(f"YOLO detected {len(results)} objects")
    return results


# ---------------------------------------------------------------------------
# ArcFace  (face embeddings)
# ---------------------------------------------------------------------------

def _preprocess_arcface(image: Image.Image, box: Optional[list[int]] = None) -> np.ndarray:
    """
    If a bounding box [x1, y1, x2, y2] is given, crop to that region first.
    Resize to 112×112, normalise to [-1, 1], return NCHW float32 tensor.
    """
    if box is not None:
        image = image.crop(tuple(box))
    image = image.resize((ARCFACE_INPUT_SIZE, ARCFACE_INPUT_SIZE), Image.BILINEAR)
    arr = np.asarray(image, dtype=np.float32)  # HWC, 0-255
    arr = (arr - 127.5) / 128.0  # normalise to ~ [-1, 1]
    arr = arr.transpose(2, 0, 1)  # CHW
    return np.expand_dims(arr, 0)  # NCHW


def process_arcface(image_bytes: bytes) -> list[dict]:
    """
    Detect faces with YOLO first (filter to 'person' class boxes as proxy,
    or use the full image if no face detector is available), then compute
    128-d/512-d face embeddings with ArcFace.
    Returns list of {embedding, box}.
    """
    session = model_manager.get("arcface")
    if session is None:
        logger.warning("ArcFace model not loaded, attempting on-the-fly load …")
        if not model_manager.load("arcface"):
            return []
        session = model_manager.get("arcface")
        if session is None:
            return []

    image = _load_image(image_bytes)

    # Detect candidate face boxes via YOLO (look for 'person' detections)
    person_boxes: list[list[int]] = []
    yolo_results = process_yolo(image_bytes)
    for det in yolo_results:
        if det["class"] == "person" and det["score"] >= 0.5:
            person_boxes.append(det["box"])

    # Fallback: entire image if no person detected
    if not person_boxes:
        person_boxes = [[0, 0, image.width, image.height]]

    input_name = session.get_inputs()[0].name
    results = []
    for box in person_boxes:
        tensor = _preprocess_arcface(image, box)
        embedding = session.run(None, {input_name: tensor})[0]
        emb_list = embedding.flatten().tolist()
        # L2-normalise
        norm = float(np.linalg.norm(emb_list))
        if norm > 0:
            emb_list = [v / norm for v in emb_list]
        results.append({
            "embedding": emb_list,
            "box": box,
        })

    logger.info(f"ArcFace produced {len(results)} face embeddings")
    return results


# ---------------------------------------------------------------------------
# CLIP  (semantic image embeddings)
# ---------------------------------------------------------------------------

def _preprocess_clip_image(image: Image.Image) -> np.ndarray:
    """
    Standard CLIP ViT-B/32 preprocessing:
      - Resize shortest side to CLIP_INPUT_SIZE, centre-crop to CLIP_INPUT_SIZE×CLIP_INPUT_SIZE
      - Normalise with ImageNet mean/std
      - Return NCHW float32
    """
    # Resize shortest side to CLIP_INPUT_SIZE
    w, h = image.size
    scale = CLIP_INPUT_SIZE / min(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    image = image.resize((new_w, new_h), Image.BICUBIC)

    # Centre crop CLIP_INPUT_SIZE×CLIP_INPUT_SIZE
    left = (new_w - CLIP_INPUT_SIZE) // 2
    top = (new_h - CLIP_INPUT_SIZE) // 2
    image = image.crop((left, top, left + CLIP_INPUT_SIZE, top + CLIP_INPUT_SIZE))

    arr = np.asarray(image, dtype=np.float32) / 255.0
    mean = np.array(CLIP_MEAN, dtype=np.float32)
    std  = np.array(CLIP_STD, dtype=np.float32)
    arr = (arr - mean) / std
    arr = arr.transpose(2, 0, 1)
    return np.expand_dims(arr, 0)


def process_clip(image_bytes: bytes) -> list[dict]:
    """
    Compute a CLIP image embedding.
    Returns [{"embedding": [...]}].
    """
    session = model_manager.get("clip_visual")
    if session is None:
        logger.warning("CLIP visual model not loaded, attempting on-the-fly load …")
        if not model_manager.load("clip_visual"):
            return []
        session = model_manager.get("clip_visual")
        if session is None:
            return []

    image = _load_image(image_bytes)
    tensor = _preprocess_clip_image(image)

    inputs = {}
    for i in session.get_inputs():
        if i.name in ["pixel_values", "input"]:
            inputs[i.name] = tensor
        elif i.name == "input_ids":
            inputs[i.name] = np.array([[49406, 49407]], dtype=np.int64)
        elif i.name == "attention_mask":
            inputs[i.name] = np.array([[1, 1]], dtype=np.int64)
            
    if not inputs and session.get_inputs():
        inputs[session.get_inputs()[0].name] = tensor

    out_names = [o.name for o in session.get_outputs()]
    fetch = ["image_embeds"] if "image_embeds" in out_names else [out_names[0]]
    outputs = session.run(fetch, inputs)
    embedding = outputs[0].flatten().tolist()

    # L2-normalise
    norm = float(np.linalg.norm(embedding))
    if norm > 0:
        embedding = [v / norm for v in embedding]

    logger.info("CLIP embedding computed")
    return [{"embedding": embedding}]


def process_clip_text(text: str) -> list[dict]:
    """
    Compute a CLIP text embedding for natural language semantic search.
    Returns [{"embedding": [...]}] where the embedding vector is 512-d.
    """
    session = model_manager.get("clip_visual")
    if session is None:
        logger.warning("CLIP model not loaded, attempting on-the-fly load …")
        if not model_manager.load("clip_visual"):
            return []
        session = model_manager.get("clip_visual")
        if session is None:
            return []

    from utils import tokenizer_manager
    tokenizer = tokenizer_manager.get()
    encoding = tokenizer.encode(text)

    # Prepare inputs (dummy pixel values satisfy the unified CLIP model signature)
    inputs = {
        "input_ids": np.array([encoding.ids], dtype=np.int64),
        "attention_mask": np.array([encoding.attention_mask], dtype=np.int64),
        "pixel_values": np.zeros((1, 3, CLIP_INPUT_SIZE, CLIP_INPUT_SIZE), dtype=np.float32),
    }

    # Run session and extract 'text_embeds'
    outputs = session.run(["text_embeds"], inputs)
    embedding = outputs[0].flatten().tolist()

    # L2-normalise for cosine-similarity comparison
    norm = float(np.linalg.norm(embedding))
    if norm > 0:
        embedding = [v / norm for v in embedding]

    logger.info(f"CLIP text embedding computed for query: '{text[:20]}...' ")
    return [{"embedding": embedding}]

