"""
ASL BiLSTM Backend Server - Complete Single File
Integrates your trained BiLSTM model with live video streaming
Everything in one file - just run and go!
"""

from flask import Flask, render_template_string
from flask_sock import Sock
import torch
import torch.nn as nn
import cv2
import mediapipe as mp
import numpy as np
import base64
import json
from collections import deque
import io
from PIL import Image

app = Flask(__name__)
sock = Sock(app)

# ============= YOUR MODEL DEFINITION =============
class ASL_BiLSTM(nn.Module):
    def __init__(
        self,
        input_size: int = 222,
        hidden_size: int = 256,  # Default from config
        output_size: int = 26,   # A-Z letters
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_size * 2, output_size)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        logits = self.fc(last_hidden)
        return logits


# ============= GLOBAL SETUP =============
# Load your trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ASL_BiLSTM()

# TODO: Load your trained weights
# model.load_state_dict(torch.load('your_model.pth', map_location=device))
model.to(device)
model.eval()

# MediaPipe setup for hand landmark detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Class labels (adjust to match your training)
CLASS_LABELS = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')  # 26 letters

# Frame buffer for sequence (60 frames)
SEQUENCE_LENGTH = 60


# ============= HELPER FUNCTIONS =============
def extract_landmarks(frame):
    """
    Extract hand landmarks from a video frame
    Returns 222 features: left hand (21*3) + right hand (21*3) + pose (33*4)
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    # Initialize feature vector with zeros
    features = np.zeros(222)
    
    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Get hand type (left or right)
            hand_type = results.multi_handedness[idx].classification[0].label
            
            # Extract 21 landmarks * 3 coordinates = 63 features per hand
            landmarks_array = []
            for landmark in hand_landmarks.landmark:
                landmarks_array.extend([landmark.x, landmark.y, landmark.z])
            
            # Place in correct position in feature vector
            if hand_type == 'Left':
                features[0:63] = landmarks_array
            else:  # Right hand
                features[63:126] = landmarks_array
    
    # Note: Remaining 96 features (126-222) are for pose landmarks
    # If you're using pose, add MediaPipe pose detection here
    
    return features


def predict_gesture(sequence_buffer):
    """
    Predict ASL gesture from a sequence of frames
    sequence_buffer: list of 60 frames, each with 222 features
    """
    if len(sequence_buffer) < SEQUENCE_LENGTH:
        return None, 0.0
    
    # Convert to tensor: (1, 60, 222)
    sequence = np.array(list(sequence_buffer))
    sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        logits = model(sequence_tensor)
        probs = torch.softmax(logits, dim=1)
        confidence, predicted = torch.max(probs, 1)
    
    predicted_class = CLASS_LABELS[predicted.item()]
    confidence_score = confidence.item()
    
    # Only return prediction if confidence is high enough
    if confidence_score > 0.7:  # Threshold
        return predicted_class, confidence_score
    
    return None, confidence_score


# ============= WEBSOCKET ENDPOINT =============
@sock.route('/video')
def video_stream(ws):
    """WebSocket endpoint for receiving video frames and sending predictions"""
    print("Client connected")
    
    # Buffer to store sequence of frames
    frame_buffer = deque(maxlen=SEQUENCE_LENGTH)
    frame_count = 0
    
    try:
        while True:
            message = ws.receive()
            if message is None:
                break
            
            data = json.loads(message)
            
            # Decode base64 image
            img_data = data['frame'].split(',')[1]
            img_bytes = base64.b64decode(img_data)
            img = Image.open(io.BytesIO(img_bytes))
            frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            
            # Extract landmarks
            features = extract_landmarks(frame)
            frame_buffer.append(features)
            frame_count += 1
            
            # Only predict every 5 frames (to reduce computation)
            if frame_count % 5 == 0 and len(frame_buffer) == SEQUENCE_LENGTH:
                predicted_letter, confidence = predict_gesture(frame_buffer)
                
                if predicted_letter:
                    response = {
                        'prediction': predicted_letter,
                        'confidence': f"{confidence:.2f}",
                        'timestamp': data.get('timestamp', 0)
                    }
                    ws.send(json.dumps(response))
    
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        print("Client disconnected")


# ============= COMPLETE HTML FRONTEND (INLINE) =============
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ASL Live Translation</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }

        .container {
            max-width: 1000px;
            width: 100%;
        }

        .header {
            text-align: center;
            color: white;
            margin-bottom: 2rem;
        }

        .header h1 {
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
            text-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
        }

        .header p {
            font-size: 1.1rem;
            opacity: 0.95;
        }

        .main-content {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 1.5rem;
        }

        .video-section {
            background: white;
            border-radius: 1rem;
            padding: 1.5rem;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
        }

        #video {
            width: 100%;
            border-radius: 0.75rem;
            background: #000;
            aspect-ratio: 4/3;
            object-fit: cover;
        }

        .controls {
            display: flex;
            gap: 1rem;
            margin-top: 1rem;
            justify-content: center;
        }

        .control-btn {
            padding: 0.75rem 1.5rem;
            border: none;
            border-radius: 0.5rem;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .btn-start {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }

        .btn-start:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
        }

        .btn-stop {
            background: #dc2626;
            color: white;
        }

        .btn-stop:hover {
            background: #b91c1c;
        }

        .btn-clear {
            background: #6b7280;
            color: white;
        }

        .btn-clear:hover {
            background: #4b5563;
        }

        .info-panel {
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
        }

        .prediction-card {
            background: white;
            border-radius: 1rem;
            padding: 2rem;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
            text-align: center;
        }

        .prediction-label {
            font-size: 0.875rem;
            color: #6b7280;
            text-transform: uppercase;
            letter-spacing: 2px;
            margin-bottom: 1rem;
        }

        .prediction-display {
            font-size: 5rem;
            font-weight: bold;
            color: #667eea;
            min-height: 100px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 1rem;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .confidence-bar {
            margin-top: 1rem;
        }

        .confidence-label {
            font-size: 0.875rem;
            color: #6b7280;
            margin-bottom: 0.5rem;
            display: flex;
            justify-content: space-between;
        }

        .progress-bar {
            width: 100%;
            height: 8px;
            background: #e5e7eb;
            border-radius: 4px;
            overflow: hidden;
        }

        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            transition: width 0.3s ease;
            width: 0%;
        }

        .word-card {
            background: white;
            border-radius: 1rem;
            padding: 1.5rem;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        }

        .word-card h3 {
            color: #667eea;
            margin-bottom: 1rem;
            font-size: 1.25rem;
        }

        .word-display {
            font-size: 1.75rem;
            color: #1f2937;
            letter-spacing: 3px;
            font-weight: 600;
            padding: 1rem;
            background: #f3f4f6;
            border-radius: 0.5rem;
            min-height: 60px;
            display: flex;
            align-items: center;
            word-break: break-all;
        }

        .word-display.empty {
            color: #9ca3af;
            font-style: italic;
            letter-spacing: normal;
        }

        .status-card {
            background: white;
            border-radius: 1rem;
            padding: 1.25rem;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        }

        .status-item {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            padding: 0.75rem;
            margin-bottom: 0.5rem;
            background: #f9fafb;
            border-radius: 0.5rem;
        }

        .status-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: #6b7280;
        }

        .status-dot.active {
            background: #10b981;
            animation: pulse 2s infinite;
        }

        .status-dot.error {
            background: #ef4444;
        }

        @keyframes pulse {
            0%, 100% {
                opacity: 1;
                transform: scale(1);
            }
            50% {
                opacity: 0.6;
                transform: scale(1.1);
            }
        }

        .status-text {
            font-size: 0.875rem;
            color: #374151;
            flex: 1;
        }

        .history-card {
            background: white;
            border-radius: 1rem;
            padding: 1.5rem;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        }

        .history-card h3 {
            color: #667eea;
            margin-bottom: 1rem;
            font-size: 1.25rem;
        }

        .history-list {
            max-height: 200px;
            overflow-y: auto;
        }

        .history-item {
            padding: 0.75rem;
            background: #f9fafb;
            border-radius: 0.5rem;
            margin-bottom: 0.5rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.875rem;
        }

        .history-letter {
            font-weight: 600;
            color: #667eea;
            font-size: 1.25rem;
        }

        .history-time {
            color: #9ca3af;
            font-size: 0.75rem;
        }

        .keyboard-hint {
            margin-top: 1rem;
            padding: 0.75rem;
            background: #fef3c7;
            border-radius: 0.5rem;
            font-size: 0.75rem;
            color: #92400e;
            text-align: center;
        }

        @media (max-width: 768px) {
            .main-content {
                grid-template-columns: 1fr;
            }

            .header h1 {
                font-size: 2rem;
            }

            .prediction-display {
                font-size: 3.5rem;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤟 ASL Live Translation</h1>
            <p>BiLSTM Neural Network • Real-time Sign Language Recognition</p>
        </div>

        <div class="main-content">
            <!-- Left Column: Video -->
            <div class="video-section">
                <video id="video" autoplay playsinline></video>
                <canvas id="canvas" style="display: none;"></canvas>

                <div class="controls">
                    <button id="startBtn" class="control-btn btn-start">
                        <span>▶</span> Start
                    </button>
                    <button id="stopBtn" class="control-btn btn-stop" style="display: none;">
                        <span>⏹</span> Stop
                    </button>
                    <button id="clearBtn" class="control-btn btn-clear">
                        <span>🗑</span> Clear Word
                    </button>
                </div>
            </div>

            <!-- Right Column: Info -->
            <div class="info-panel">
                <!-- Current Prediction -->
                <div class="prediction-card">
                    <div class="prediction-label">Current Sign</div>
                    <div class="prediction-display" id="prediction">-</div>
                    <div class="confidence-bar">
                        <div class="confidence-label">
                            <span>Confidence</span>
                            <span id="confidenceText">0%</span>
                        </div>
                        <div class="progress-bar">
                            <div class="progress-fill" id="confidenceFill"></div>
                        </div>
                    </div>
                </div>

                <!-- Spelled Word -->
                <div class="word-card">
                    <h3>Spelled Word</h3>
                    <div class="word-display" id="word">(empty)</div>
                    <div class="keyboard-hint">
                        💡 Press BACKSPACE to delete • ESC to clear all
                    </div>
                </div>

                <!-- Status -->
                <div class="status-card">
                    <div class="status-item">
                        <div class="status-dot" id="cameraStatus"></div>
                        <div class="status-text" id="cameraText">Camera: Initializing...</div>
                    </div>
                    <div class="status-item">
                        <div class="status-dot" id="wsStatus"></div>
                        <div class="status-text" id="wsText">Backend: Disconnected</div>
                    </div>
                    <div class="status-item">
                        <div class="status-dot" id="modelStatus"></div>
                        <div class="status-text" id="modelText">Model: Waiting...</div>
                    </div>
                </div>

                <!-- History -->
                <div class="history-card">
                    <h3>Recent Detections</h3>
                    <div class="history-list" id="historyList">
                        <div style="text-align: center; color: #9ca3af; padding: 2rem;">
                            No detections yet
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // DOM Elements
        const video = document.getElementById('video');
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        const predictionEl = document.getElementById('prediction');
        const confidenceTextEl = document.getElementById('confidenceText');
        const confidenceFillEl = document.getElementById('confidenceFill');
        const wordEl = document.getElementById('word');
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');
        const clearBtn = document.getElementById('clearBtn');
        const historyListEl = document.getElementById('historyList');

        // Status elements
        const cameraStatus = document.getElementById('cameraStatus');
        const cameraText = document.getElementById('cameraText');
        const wsStatus = document.getElementById('wsStatus');
        const wsText = document.getElementById('wsText');
        const modelStatus = document.getElementById('modelStatus');
        const modelText = document.getElementById('modelText');

        // State
        let ws = null;
        let word = '';
        let isRunning = false;
        let sendFrameInterval = null;
        let lastPrediction = '';
        let predictionCount = 0;
        let detectionHistory = [];

        // Configuration
        const WS_URL = 'ws://' + window.location.host + '/video';
        const FRAME_INTERVAL = 100; // Send frame every 100ms
        const REQUIRED_PREDICTIONS = 3; // Need 3 consistent predictions

        // Initialize
        function init() {
            setupCamera();
            setupEventListeners();
        }

        // Setup camera
        async function setupCamera() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({
                    video: {
                        width: { ideal: 640 },
                        height: { ideal: 480 },
                        facingMode: 'user'
                    }
                });
                
                video.srcObject = stream;
                cameraStatus.classList.add('active');
                cameraText.textContent = 'Camera: Ready';
            } catch (err) {
                console.error('Camera error:', err);
                cameraStatus.classList.add('error');
                cameraText.textContent = 'Camera: Access Denied';
                alert('Please allow camera access to use ASL translation');
            }
        }

        // Setup event listeners
        function setupEventListeners() {
            startBtn.addEventListener('click', startTranslation);
            stopBtn.addEventListener('click', stopTranslation);
            clearBtn.addEventListener('click', clearWord);

            // Keyboard shortcuts
            document.addEventListener('keydown', (e) => {
                if (e.key === 'Backspace') {
                    e.preventDefault();
                    deleteLastLetter();
                } else if (e.key === 'Escape') {
                    clearWord();
                } else if (e.key === ' ' && !isRunning) {
                    e.preventDefault();
                    startTranslation();
                }
            });
        }

        // Start translation
        function startTranslation() {
            if (isRunning) return;

            isRunning = true;
            startBtn.style.display = 'none';
            stopBtn.style.display = 'flex';

            connectWebSocket();
        }

        // Stop translation
        function stopTranslation() {
            if (!isRunning) return;

            isRunning = false;
            stopBtn.style.display = 'none';
            startBtn.style.display = 'flex';

            if (ws) {
                ws.close();
                ws = null;
            }

            if (sendFrameInterval) {
                clearInterval(sendFrameInterval);
                sendFrameInterval = null;
            }

            updateStatus('disconnected');
        }

        // Connect to WebSocket
        function connectWebSocket() {
            wsText.textContent = 'Backend: Connecting...';

            ws = new WebSocket(WS_URL);

            ws.onopen = () => {
                console.log('WebSocket connected');
                wsStatus.classList.add('active');
                wsText.textContent = 'Backend: Connected';
                modelStatus.classList.add('active');
                modelText.textContent = 'Model: Active';
                startSendingFrames();
            };

            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                handlePrediction(data);
            };

            ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                wsStatus.classList.add('error');
                wsText.textContent = 'Backend: Error';
                modelStatus.classList.remove('active');
                modelText.textContent = 'Model: Inactive';
            };

            ws.onclose = () => {
                console.log('WebSocket closed');
                updateStatus('disconnected');
            };
        }

        // Start sending frames
        function startSendingFrames() {
            sendFrameInterval = setInterval(() => {
                if (ws && ws.readyState === WebSocket.OPEN) {
                    sendFrame();
                }
            }, FRAME_INTERVAL);
        }

        // Send frame to backend
        function sendFrame() {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            ctx.drawImage(video, 0, 0);

            const frameData = canvas.toDataURL('image/jpeg', 0.8);
            
            ws.send(JSON.stringify({
                frame: frameData,
                timestamp: Date.now()
            }));
        }

        // Handle prediction from backend
        function handlePrediction(data) {
            const { prediction, confidence } = data;
            
            // Update display
            predictionEl.textContent = prediction;
            
            // Update confidence bar
            const confidencePercent = Math.round(parseFloat(confidence) * 100);
            confidenceTextEl.textContent = confidencePercent + '%';
            confidenceFillEl.style.width = confidencePercent + '%';

            // Add to history
            addToHistory(prediction, confidence);

            // Build word from consistent predictions
            if (prediction === lastPrediction) {
                predictionCount++;
                if (predictionCount >= REQUIRED_PREDICTIONS) {
                    addLetterToWord(prediction);
                    predictionCount = 0;
                }
            } else {
                lastPrediction = prediction;
                predictionCount = 1;
            }
        }

        // Add letter to word
        function addLetterToWord(letter) {
            word += letter;
            updateWordDisplay();
        }

        // Delete last letter
        function deleteLastLetter() {
            if (word.length > 0) {
                word = word.slice(0, -1);
                updateWordDisplay();
            }
        }

        // Clear word
        function clearWord() {
            word = '';
            updateWordDisplay();
        }

        // Update word display
        function updateWordDisplay() {
            if (word.length === 0) {
                wordEl.textContent = '(empty)';
                wordEl.classList.add('empty');
            } else {
                wordEl.textContent = word;
                wordEl.classList.remove('empty');
            }
        }

        // Add to history
        function addToHistory(letter, confidence) {
            const now = new Date();
            const timeStr = now.toLocaleTimeString('en-US', { 
                hour: '2-digit', 
                minute: '2-digit', 
                second: '2-digit' 
            });

            detectionHistory.unshift({
                letter: letter,
                confidence: confidence,
                time: timeStr
            });

            // Keep only last 10
            if (detectionHistory.length > 10) {
                detectionHistory.pop();
            }

            updateHistoryDisplay();
        }

        // Update history display
        function updateHistoryDisplay() {
            if (detectionHistory.length === 0) {
                historyListEl.innerHTML = '<div style="text-align: center; color: #9ca3af; padding: 2rem;">No detections yet</div>';
                return;
            }

            historyListEl.innerHTML = detectionHistory.map(item => `
                <div class="history-item">
                    <span class="history-letter">${item.letter}</span>
                    <span>${Math.round(parseFloat(item.confidence) * 100)}%</span>
                    <span class="history-time">${item.time}</span>
                </div>
            `).join('');
        }

        // Update status
        function updateStatus(status) {
            if (status === 'disconnected') {
                wsStatus.classList.remove('active');
                wsStatus.classList.remove('error');
                wsText.textContent = 'Backend: Disconnected';
                modelStatus.classList.remove('active');
                modelText.textContent = 'Model: Inactive';
            }
        }

        // Initialize on load
        window.addEventListener('load', init);
    </script>
</body>
</html>
"""


# ============= FLASK ROUTES =============
@app.route('/')
def index():
    """Serve the complete HTML interface"""
    return render_template_string(HTML_TEMPLATE)


@app.route('/health')
def health():
    """Health check endpoint"""
    return {
        'status': 'healthy', 
        'model': 'ASL_BiLSTM', 
        'device': str(device),
        'classes': len(CLASS_LABELS)
    }


# ============= RUN SERVER =============
if __name__ == '__main__':
    print("=" * 60)
    print(" ASL BiLSTM Translation Server - Complete Single File")
    print("=" * 60)
    print(f"🖥️  Device: {device}")
    print(f"🧠 Model: ASL_BiLSTM")
    print(f"📊 Sequence length: {SEQUENCE_LENGTH} frames")
    print(f"🔤 Classes: {len(CLASS_LABELS)}")
    print("=" * 60)
    print(f"\n✨ Server starting on http://localhost:5000")
    print(f"✨ Also available at http://127.0.0.1:5000")
    print(f"\n📝 TODO: Load your trained model weights!")
    print(f"   Uncomment line: model.load_state_dict(...)")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5000, debug=True)