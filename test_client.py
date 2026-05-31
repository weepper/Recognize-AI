"""
Functional test client for Recognize ExApp.

Requirements:
    pip install requests

Usage:
    1. Start the server:  python main.py
    2. Run this script:   python test_client.py
"""
import sys
import json
import requests

BASE_URL = "http://127.0.0.1:8000"


def test_health():
    """Test the /health endpoint."""
    print("--- Testing Health Check ---")
    res = requests.get(f"{BASE_URL}/health")
    print(f"Status: {res.status_code}")
    print(f"Response: {json.dumps(res.json(), indent=2)}\n")
    assert res.status_code == 200, f"Health check failed with status {res.status_code}"


def test_models_status():
    """Test the /models/status endpoint."""
    print("--- Testing Model Status ---")
    res = requests.get(f"{BASE_URL}/models/status")
    print(f"Status: {res.status_code}")
    print(f"Response: {json.dumps(res.json(), indent=2)}\n")
    assert res.status_code == 200, f"Model status failed with status {res.status_code}"


def test_yolo(image_data: bytes):
    """Test the /analyze/objects endpoint."""
    print("--- Testing YOLO Object Detection ---")
    res = requests.post(
        f"{BASE_URL}/analyze/objects",
        files={"file": ("test.jpg", image_data, "image/jpeg")},
    )
    print(f"Status: {res.status_code}")
    print(f"Response:\n{json.dumps(res.json(), indent=2)}\n")


def test_arcface(image_data: bytes):
    """Test the /analyze/faces endpoint."""
    print("--- Testing ArcFace Face Recognition ---")
    res = requests.post(
        f"{BASE_URL}/analyze/faces",
        files={"file": ("test.jpg", image_data, "image/jpeg")},
    )
    print(f"Status: {res.status_code}")
    print(f"Response:\n{json.dumps(res.json(), indent=2)}\n")


def test_clip(image_data: bytes) -> list[float]:
    """Test the /analyze/semantic endpoint."""
    print("--- Testing CLIP Semantic Search ---")
    res = requests.post(
        f"{BASE_URL}/analyze/semantic",
        files={"file": ("test.jpg", image_data, "image/jpeg")},
    )
    print(f"Status: {res.status_code}")
    clip_res = res.json()
    if isinstance(clip_res, list) and len(clip_res) > 0:
        emb = clip_res[0].get("embedding", [])
        print(f"Response: Embedding vector length = {len(emb)}\n")
        return emb
    else:
        print(f"Response:\n{json.dumps(clip_res, indent=2)}\n")
        return []


def test_clip_text(query: str) -> list[float]:
    """Test the /analyze/text endpoint for a given query."""
    print(f"--- Testing CLIP Text Embedding for: '{query}' ---")
    res = requests.post(
        f"{BASE_URL}/analyze/text",
        json={"text": query},
    )
    print(f"Status: {res.status_code}")
    res_json = res.json()
    if isinstance(res_json, list) and len(res_json) > 0:
        emb = res_json[0].get("embedding", [])
        print(f"Response: Embedding vector length = {len(emb)}\n")
        return emb
    else:
        print(f"Response:\n{json.dumps(res_json, indent=2)}\n")
        return []


def test_semantic_verification(image_emb: list[float]):
    """Verify semantic matching using cosine similarity (dot product of L2-normalized embeddings)."""
    print("--- Testing Semantic Cosine Similarity Verification ---")
    if not image_emb:
        print("Skipping semantic verification: Image embedding is empty.\n")
        return

    # Let's query a matching text and a mismatched text
    matching_query = "a photo of a black puppy dog"
    mismatched_query = "a high-speed formula 1 racecar on a track"

    match_emb = test_clip_text(matching_query)
    mismatch_emb = test_clip_text(mismatched_query)

    if not match_emb or not mismatch_emb:
        print("Skipping semantic verification: Text embeddings are empty.\n")
        return

    # Cosine similarity for L2-normalized vectors is just the dot product
    sim_match = sum(a * b for a, b in zip(image_emb, match_emb))
    sim_mismatch = sum(a * b for a, b in zip(image_emb, mismatch_emb))

    print(f"Similarity with '{matching_query}': {sim_match:.4f}")
    print(f"Similarity with '{mismatched_query}': {sim_mismatch:.4f}")

    # Assertion check
    assert sim_match > sim_mismatch, (
        f"Semantic search failed! Matching query similarity ({sim_match:.4f}) "
        f"must be higher than mismatched query similarity ({sim_mismatch:.4f})."
    )
    print("Semantic search similarity verification PASSED successfully!\n")


def main():
    # 0. Check server connectivity
    print(f"Connecting to {BASE_URL} ...\n")
    try:
        requests.get(f"{BASE_URL}/health", timeout=5)
    except requests.ConnectionError:
        print(f"ERROR: Cannot connect to {BASE_URL}.")
        print("Make sure the server is running: python main.py")
        sys.exit(1)

    # 1. Test metadata endpoints
    test_health()
    test_models_status()

    # 2. Download a sample image (Picsum ID 237 is a lovely black puppy dog)
    # To use a local picture instead, uncomment the following two lines:
    # with open("my_local_picture.jpg", "rb") as f:
    #     image_data = f.read()
    IMAGE_URL = "https://picsum.photos/id/237/800/600.jpg"
    print(f"Downloading test image (puppy dog) from {IMAGE_URL}...")
    response = requests.get(IMAGE_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    response.raise_for_status()
    image_data = response.content
    print(f"Image loaded successfully ({len(image_data)} bytes).\n")

    # 3. Test analysis endpoints
    test_yolo(image_data)
    test_arcface(image_data)
    image_emb = test_clip(image_data)

    # 4. Semantic Verification check
    test_semantic_verification(image_emb)

    print("=" * 50)
    print("All tests completed!")


if __name__ == "__main__":
    main()
