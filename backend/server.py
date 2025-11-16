# backend/server.py
import base64
import io
import os
from typing import List, Optional

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# -------------------------------------------------
# Device
# -------------------------------------------------
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using device: {device}")

# -------------------------------------------------
# TODO: YOUR MODEL CLASS HERE
# -------------------------------------------------
# Replace this stub with your actual BiLSTM model class.
# Make sure its __init__ signature matches how you trained it.
class ASLBiLSTM(torch.nn.Module):
    def __init__(self, input_dim: int = 222, hidden_dim: int = 256,
                 num_layers: int = 2, num_classes: int = 6):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        self.fc = torch.nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        # x: (B, T, F)
        out, _ = self.lstm(x)        # (B, T, 2H)
        pooled = out.mean(dim=1)     # temporal average pooling
        logits = self.fc(pooled)     # (B, C)
        return logits


# -------------------------------------------------
# TODO: labels in the same order as training
# -------------------------------------------------
# For example, if you trained on:
# 0:"A", 1:"B", 2:"C", 3:"D", 4:"E", 5:"F"
LABELS: List[str] = ["A", "B", "C", "D", "E", "F"]  # <-- change to your labels


# -------------------------------------------------
# Load model + weights
# -------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best.pt")

model = ASLBiLSTM(
    input_dim=222,
    hidden_dim=256,
    num_layers=2,
    num_classes=len(LABELS),
).to(device)

if os.path.exists(MODEL_PATH):
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print("Loaded model weights from best.pt")
else:
    print("WARNING: best.pt not found. Model will use random weights.")


# -------------------------------------------------
# TODO: Frame → keypoints → sequence tensor
# -------------------------------------------------
# You must implement this using your MediaPipe + preprocessing pipeline
# so that it returns a tensor of shape (T=60, F=222) in float32.
def preprocess_frame_to_sequence(img: Image.Image) -> Optional[torch.Tensor]:
    """
    img: PIL Image (RGB)
    returns: torch.FloatTensor of shape (60, 222) or None if no hand detected
    """
    # ----- EXAMPLE DUMMY IMPLEMENTATION -----
    # Replace everything in this function with your real keypoint code.

    # Example: resize & flatten just to have something structurally valid
    img = img.resize((64, 64))
    arr = np.array(img).astype(np.float32) / 255.0  # (64,64,3)
    flat = arr.reshape(-1)                          # (12288,)

    # For demo purposes, pad/truncate to 60*222 = 13320
    target_len = 60 * 222
    if flat.shape[0] < target_len:
        pad = np.zeros(target_len - flat.shape[0], dtype=np.float32)
        flat = np.concatenate([flat, pad])
    elif flat.shape[0] > target_len:
        flat = flat[:target_len]

    seq = flat.reshape(60, 222)  # (T,F)

    # Normalize per-sample
    mean = seq.mean()
    std = seq.std() + 1e-6
    seq = (seq - mean) / std

    return torch.from_numpy(seq)  # (60,222)


# -------------------------------------------------
# FastAPI app + CORS
# -------------------------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # you can lock this down later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"status": "ok", "message": "ASL BiLSTM backend running"}


# -------------------------------------------------
# /video WebSocket endpoint
# -------------------------------------------------
@app.websocket("/video")
async def video_ws(ws: WebSocket):
    await ws.accept()
    print("WebSocket client connected")

    try:
        while True:
            msg = await ws.receive_text()
            # Expect JSON: {"frame": "data:image/jpeg;base64,...", "timestamp": ...}
            import json
            data = json.loads(msg)
            frame_data_url = data.get("frame")

            if not frame_data_url or not frame_data_url.startswith("data:image"):
                continue

            # Strip header "data:image/jpeg;base64,"
            header, b64data = frame_data_url.split(",", 1)
            img_bytes = base64.b64decode(b64data)

            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

            seq = preprocess_frame_to_sequence(img)
            if seq is None:
                # no detection, could send a "no hand" message if you want
                continue

            seq = seq.to(device=device, dtype=torch.float32).unsqueeze(0)  # (1,60,222)

            with torch.no_grad():
                logits = model(seq)          # (1,C)
                probs = F.softmax(logits, dim=1)
                conf, pred_idx = torch.max(probs, dim=1)

            pred_idx = int(pred_idx.item())
            conf = float(conf.item())
            label = LABELS[pred_idx]

            out = {"prediction": label, "confidence": conf}
            await ws.send_text(json.dumps(out))

    except WebSocketDisconnect:
        print("WebSocket client disconnected")
    except Exception as e:
        print("WebSocket exception:", e)


if __name__ == "__main__":
    # Run locally: python server.py
    uvicorn.run(app, host="0.0.0.0", port=8000)