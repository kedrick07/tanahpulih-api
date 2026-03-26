# app.py
import json, io
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from fastapi import FastAPI, UploadFile, File

app = FastAPI()
DEVICE = "cpu"  # Railway has no GPU

# Load class names saved during training
with open("models/class_names.json") as f:
    CLASS_NAMES = json.load(f)
NUM_CLASSES = len(CLASS_NAMES)

# Rebuild the exact same model architecture from your train.py
def build_model(num_classes):
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, num_classes)
    )
    return model

model = build_model(NUM_CLASSES)
model.load_state_dict(torch.load("models/best_model.pth", map_location=DEVICE))
model.eval()

# Same val_transforms from your train.py
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

@app.get("/")
def root():
    return {"status": "TanahPulih CV model is running 🌾"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_idx = probs.argmax().item()
    return {
        "disease": CLASS_NAMES[pred_idx],
        "confidence": round(probs[pred_idx].item(), 4),
        "all_scores": {CLASS_NAMES[i]: round(p.item(), 4) for i, p in enumerate(probs)}
    }