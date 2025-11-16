# backend/server.py
import base64
import io
import os
import json

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import cv2

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import model
from model import ASL_BiLSTM

# Import preprocess globals
import preprocess_media as preprocess

# -------------------------------------------------
# Configuration
# -------------------------------------------------
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

print(f"🚀 Using device: {device}")

LABELS = ['hello', 'how are you', 'nice to meet you', 'please', 'sorry', 'thank you']
SEQ_LEN = 60  # Exactly 60 frames (2 seconds at 30fps)
FEATURE_DIM = 222

# -------------------------------------------------
# Load Model
# -------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../src/best.pt")

model = ASL_BiLSTM()

if os.path.exists(MODEL_PATH):
    try:
        state = torch.load(MODEL_PATH, map_location=device, weights_only=True)
        model.load_state_dict(state)
        print("✅ Loaded model weights from best.pt")
    except Exception as e:
        print(f"⚠️  Error loading model: {e}")
        print("⚠️  Using random weights")
else:
    print("⚠️  WARNING: best.pt not found - using random weights")

model.to(device)
model.eval()

# -------------------------------------------------
# Keypoint Extraction
# -------------------------------------------------
def extract_keypoints_from_frame(frame_bgr: np.ndarray):
    """Extract 222-dim keypoints from a single frame."""
    try:
        # Convert to RGB
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Process with MediaPipe
        hand_results = preprocess.hands.process(rgb)
        face_results = preprocess.face.process(rgb)

        keypoints = []

        # Hand landmarks (21 points × 3 = 63 dims)
        if hand_results.multi_hand_landmarks:
            hand = hand_results.multi_hand_landmarks[0]
            for lm in hand.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])
        else:
            keypoints.extend([0.0] * 63)

        # Face landmarks (46 points × 3 = 138 dims)
        if face_results.multi_face_landmarks:
            face_lm = face_results.multi_face_landmarks[0]
            for idx in preprocess.SELECTED_FACE_LANDMARKS:
                lm = face_lm.landmark[idx]
                keypoints.extend([lm.x, lm.y, lm.z])
        else:
            keypoints.extend([0.0] * 138)

        # Head movement landmarks (7 points × 3 = 21 dims)
        if face_results.multi_face_landmarks:
            face_lm = face_results.multi_face_landmarks[0]
            for idx in preprocess.HEAD_IDX:
                lm = face_lm.landmark[idx]
                keypoints.extend([lm.x, lm.y, lm.z])
        else:
            keypoints.extend([0.0] * 21)

        return np.array(keypoints, dtype=np.float32)
    
    except Exception as e:
        print(f"Error extracting keypoints: {e}")
        # Return zero vector on error
        return np.zeros(FEATURE_DIM, dtype=np.float32)


# -------------------------------------------------
# FastAPI App
# -------------------------------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "status": "ok", 
        "message": "ASL Recognition Backend Running",
        "device": str(device),
        "labels": LABELS
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": MODEL_PATH and os.path.exists(MODEL_PATH),
        "device": str(device)
    }

# -------------------------------------------------
# WebSocket Endpoint
# -------------------------------------------------
@app.websocket("/video")
async def video_ws(ws: WebSocket):
    await ws.accept()
    print("🔌 WebSocket connected")

    try:
        while True:
            # Receive batch of frames
            msg = await ws.receive_text()
            
            try:
                data = json.loads(msg)
            except json.JSONDecodeError as e:
                print(f"❌ JSON decode error: {e}")
                await ws.send_text(json.dumps({
                    "error": "Invalid JSON format"
                }))
                continue

            frames = data.get("frames")
            
            if not isinstance(frames, list) or len(frames) == 0:
                print("❌ No frames received")
                await ws.send_text(json.dumps({
                    "error": "Expected non-empty list of frames"
                }))
                continue

            print(f"📹 Processing {len(frames)} frames...")

            key_seq = []

            # Process each frame
            for i, frame_data_url in enumerate(frames):
                try:
                    # Handle data URL format
                    if "," in frame_data_url:
                        _, b64data = frame_data_url.split(",", 1)
                    else:
                        b64data = frame_data_url

                    # Decode base64
                    img_bytes = base64.b64decode(b64data)
                    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    frame_np = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

                    # Extract keypoints
                    kp = extract_keypoints_from_frame(frame_np)
                    key_seq.append(kp)

                except Exception as e:
                    print(f"⚠️  Error processing frame {i}: {e}")
                    # Continue processing other frames

            if len(key_seq) == 0:
                print("❌ No keypoints extracted from any frame")
                await ws.send_text(json.dumps({
                    "error": "Could not extract keypoints from frames"
                }))
                continue

            print(f"✅ Extracted keypoints from {len(key_seq)} frames")

            # Normalize to exactly 60 frames for the model
            if len(key_seq) < SEQ_LEN:
                pad_count = SEQ_LEN - len(key_seq)
                pad = [np.zeros(FEATURE_DIM, dtype=np.float32)] * pad_count
                key_seq.extend(pad)
                print(f"📝 Padded sequence with {pad_count} zero frames to reach 60")
            elif len(key_seq) > SEQ_LEN:
                # Take the last 60 frames (most recent)
                key_seq = key_seq[-SEQ_LEN:]
                print(f"✂️  Trimmed sequence to last {SEQ_LEN} frames")
            else:
                print(f"✓ Perfect: Exactly {SEQ_LEN} frames")

            # Stack into array
            sequence = np.stack(key_seq)
            print(f"📊 Sequence shape: {sequence.shape}")

            # Convert to tensor
            X = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(device)

            # Run inference
            try:
                with torch.no_grad():
                    logits = model(X)
                    probs = F.softmax(logits, dim=1)
                    conf, idx = torch.max(probs, dim=1)

                label = LABELS[int(idx.item())]
                confidence = float(conf.item())

                print(f"🎯 Prediction: {label} ({confidence*100:.1f}%)")

                # Send result
                await ws.send_text(json.dumps({
                    "prediction": label,
                    "confidence": confidence
                }))

            except Exception as e:
                print(f"❌ Inference error: {e}")
                await ws.send_text(json.dumps({
                    "error": f"Model inference failed: {str(e)}"
                }))

    except WebSocketDisconnect:
        print("🔌 WebSocket disconnected")
    except Exception as e:
        print(f"❌ WebSocket error: {e}")
        try:
            await ws.send_text(json.dumps({
                "error": f"Server error: {str(e)}"
            }))
        except:
            pass


if __name__ == "__main__":
    print("🚀 Starting ASL Recognition Server...")
    print(f"📍 Device: {device}")
    print(f"📋 Labels: {LABELS}")
    uvicorn.run(app, host="0.0.0.0", port=8000)