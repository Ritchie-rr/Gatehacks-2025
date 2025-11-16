# backend/server.py
import base64
import io
import os
import json
import tempfile

import numpy as np

import torch
import torch.nn.functional as F

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import model
from model import ASL_BiLSTM

# Import preprocess_media - use its extract_keypoints_from_video function
import preprocess_media


device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

LABELS = ['hello', 'how are you', 'nice to meet you', 'please', 'sorry', 'thank you']
SEQ_LEN = 60  # Exactly 60 frames for model
FEATURE_DIM = 63


MODEL_PATH = os.path.join(os.path.dirname(__file__), "../src/best.pt")
model = ASL_BiLSTM()
model.to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()


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
        "labels": LABELS,
        "input_format": "2-second MP4/WebM video as base64"
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": MODEL_PATH and os.path.exists(MODEL_PATH),
        "device": str(device)
    }


@app.websocket("/video")
async def video_ws(ws: WebSocket):
    await ws.accept()
    print("WebSocket connected")
    
    try:
        while True:
            # Receive message
            msg = await ws.receive_text()
            
            try:
                data = json.loads(msg)
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
                await ws.send_text(json.dumps({
                    "error": "Invalid JSON format"
                }))
                continue
            
            video_base64 = data.get("video")
            video_format = data.get("format", "webm")
            
            if not video_base64:
                print("No video data received")
                await ws.send_text(json.dumps({
                    "error": "No video data in request"
                }))
                continue
            
            print(f"Received video: {len(video_base64)} chars, format: {video_format}")
            
            # Decode base64 video
            try:
                video_bytes = base64.b64decode(video_base64)
                print(f"Decoded video: {len(video_bytes)} bytes")
            except Exception as e:
                print(f"Base64 decode error: {e}")
                await ws.send_text(json.dumps({
                    "error": "Failed to decode video"
                }))
                continue
            
            # Save to temporary file
            temp_video = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{video_format}') as f:
                    f.write(video_bytes)
                    temp_video = f.name
                
                print(f"Saved to temp file: {temp_video}")
                
                # Use preprocess_media's extract_keypoints_from_video function
                # This returns a (60, 63) numpy array - already padded/trimmed to 60 frames
                sequence = preprocess_media.extract_keypoints_from_video(temp_video)
                
                print(f"Extracted keypoint sequence: {sequence.shape}")
                
                # Verify shape
                if sequence.shape != (SEQ_LEN, FEATURE_DIM):
                    raise Exception(f"Expected shape ({SEQ_LEN}, {FEATURE_DIM}), got {sequence.shape}")
                
                # Convert to tensor
                X = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(device)
                
                # Run inference
                with torch.no_grad():
                    logits = model(X)
                    probs = F.softmax(logits, dim=1)
                    conf, idx = torch.max(probs, dim=1)
                
                label = LABELS[int(idx.item())]
                confidence = float(conf.item())
                
                print(f"Prediction: {label} ({confidence*100:.1f}%)")
                
                # Send result
                await ws.send_text(json.dumps({
                    "prediction": label,
                    "confidence": confidence
                }))
                
            except Exception as e:
                print(f"Processing error: {e}")
                await ws.send_text(json.dumps({
                    "error": f"Video processing failed: {str(e)}"
                }))
            
            finally:
                # Clean up temp file
                if temp_video and os.path.exists(temp_video):
                    try:
                        os.unlink(temp_video)
                        print(f"Deleted temp file: {temp_video}")
                    except:
                        pass
    
    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")


if __name__ == "__main__":
    print("Starting ASL Recognition Server...")
    print(f"Device: {device}")
    print(f"Labels: {LABELS}")
    print(f"Input: 2-second video (WebM/MP4)")
    uvicorn.run(app, host="0.0.0.0", port=8000)