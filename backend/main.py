from pathlib import Path
from io import BytesIO
from functools import lru_cache
import time

import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / 'models' / 'defectguard_resnet18.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_SIZE = 8 * 1024 * 1024

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])
app = FastAPI(title='DefectGuard AI API', version='1.0.0')
# Vite uses its own proxy on development, but this supports local direct clients.
app.add_middleware(CORSMiddleware, allow_origins=['http://localhost:5173','http://127.0.0.1:5173'], allow_methods=['GET','POST'], allow_headers=['*'])

@lru_cache(maxsize=1)
def get_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f'Missing model checkpoint: {MODEL_PATH}')
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    classes = checkpoint['classes']
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE).eval()
    return model, classes

@app.get('/api/health')
def health():
    return {'status': 'ready' if MODEL_PATH.exists() else 'checkpoint_missing', 'device': str(DEVICE), 'model': 'ResNet18', 'checkpoint_available': MODEL_PATH.exists()}

@app.post('/api/predict')
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ('image/jpeg','image/png','image/webp'):
        raise HTTPException(400, 'Upload a JPG, PNG or WEBP image.')
    content = await file.read(MAX_SIZE + 1)
    if len(content) > MAX_SIZE:
        raise HTTPException(413, 'Image exceeds 8 MB limit.')
    try:
        image = Image.open(BytesIO(content)).convert('RGB')
        model, classes = get_model()
        tensor = transform(image).unsqueeze(0).to(DEVICE)
        start = time.perf_counter()
        with torch.inference_mode():
            probabilities = torch.softmax(model(tensor),dim=1)[0]
        if DEVICE.type == 'cuda':
            torch.cuda.synchronize()
        inference_ms = round((time.perf_counter()-start)*1000,1)
        probs = probabilities.cpu().tolist()
        idx = max(range(len(probs)), key=lambda i: probs[i])
        return {
            'prediction': classes[idx], 'confidence': probs[idx],
            'probabilities': [{'name': name, 'value': float(probs[i])} for i,name in enumerate(classes)],
            'inference_ms': inference_ms, 'device': str(DEVICE),
            'model': 'ResNet18', 'image_width': image.width, 'image_height': image.height,
        }
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc))
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(400,'Cannot read this image.')
