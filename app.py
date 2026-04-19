from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import numpy as np
import io
from PIL import Image

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)

# class_id: 0 = plastik, 1 = kaca (sesuai training-mu)
CLS_PLASTIC = 0
CLS_GLASS   = 1
CONF_THRESH = 0.5

# Load model .pt sekali saat startup
model = YOLO("botol.pt")

@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    results = model(img, verbose=False)[0]  # inference langsung ke PIL Image

    best_plastic = 0.0
    best_glass   = 0.0

    if results.boxes is not None:
        for box in results.boxes:
            conf = float(box.conf[0])
            cls  = int(box.cls[0])
            if conf < CONF_THRESH:
                continue
            if cls == CLS_PLASTIC and conf > best_plastic:
                best_plastic = conf
            if cls == CLS_GLASS and conf > best_glass:
                best_glass = conf

    return {"plastic": round(best_plastic, 4), "glass": round(best_glass, 4)}
