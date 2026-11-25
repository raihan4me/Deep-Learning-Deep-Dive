"""Fashion-MNIST inference API backed by ONNX Runtime."""
import os

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_PATH = os.path.expanduser("~/dl-lab/models/cnn.onnx")
MEAN, STD = 0.2860, 0.3530          # must match training-time preprocessing exactly
CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

app = FastAPI(title="fashion-cnn", version="1.0")
sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])


class PredictRequest(BaseModel):
    pixels: list[float]             # 784 raw grayscale values, row-major, range 0-255


@app.get("/health")
def health():
    return {"status": "ok", "model": os.path.basename(MODEL_PATH),
            "input": sess.get_inputs()[0].name}


@app.post("/predict")
def predict(req: PredictRequest):
    if len(req.pixels) != 784:
        raise HTTPException(status_code=422, detail=f"expected 784 pixel values, got {len(req.pixels)}")
    x = np.asarray(req.pixels, dtype=np.float32)
    if x.min() < 0 or x.max() > 255:
        raise HTTPException(status_code=422, detail="pixel values must be in [0, 255]")
    x = ((x / 255.0 - MEAN) / STD).reshape(1, 1, 28, 28).astype(np.float32)
    logits = sess.run(None, {"input": x})[0][0]
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    top3 = np.argsort(probs)[::-1][:3]
    idx = int(top3[0])
    return {"class_id": idx, "class_name": CLASSES[idx],
            "confidence": round(float(probs[idx]), 4),
            "top3": [{"class_name": CLASSES[int(i)], "prob": round(float(probs[i]), 4)} for i in top3]}
