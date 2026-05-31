from optimum.onnxruntime import ORTModelForFeatureExtraction

model_id = "openai/clip-vit-base-patch32"
print(f"Exporting {model_id} to ONNX locally...")

# This automatically downloads PyTorch weights and exports to ONNX
model = ORTModelForFeatureExtraction.from_pretrained(model_id, export=True)
model.save_pretrained("clip_export_api")

import shutil
shutil.copy("clip_export_api/vision_model.onnx", "models/clip_visual.onnx")
print("Export complete. Vision model saved to models/clip_visual.onnx")
