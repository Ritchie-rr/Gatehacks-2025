import cv2
import torch
import numpy as np
import mediapipe as mp
from model import ASL_BiLSTM
from config import CLASSES
import torch.nn.functional as F

# ---------------- Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------------- Load trained model
model = ASL_BiLSTM().to(device)
model.load_state_dict(torch.load("best.pt", map_location=device))
model.eval()

# ---------------- MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# ---------------- Detect SEQ_LEN from dataloader
dm = ASLDataModule()
dm.setup()
train_loader = dm.train_dataloader()
for x, _ in train_loader:
    SEQ_LEN = x.shape[1]  # second dimension = sequence length
    print("Detected SEQ_LEN:", SEQ_LEN)
    break

# ---------------- LSTM frame buffer
frame_buffer = []

def preprocess_frame(landmarks):
    """
    Convert single frame's hand landmarks into a tensor sequence for LSTM
    """
    global frame_buffer
    
    # Normalize landmarks
    landmarks = (landmarks - np.mean(landmarks)) / (np.std(landmarks) + 1e-6)
    
    # Add to buffer
    frame_buffer.append(landmarks)
    
    # Keep only last SEQ_LEN frames
    if len(frame_buffer) > SEQ_LEN:
        frame_buffer.pop(0)
    
    # Pad with zeros if not enough frames yet
    while len(frame_buffer) < SEQ_LEN:
        frame_buffer.insert(0, np.zeros_like(landmarks))
    
    seq = np.array(frame_buffer)  # shape: (SEQ_LEN, 63)
    seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)  # (1, SEQ_LEN, 63)
    
    return seq_tensor

# ---------------- Webcam capture
cap = cv2.VideoCapture(0)
print("Starting live ASL gesture detection. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Convert landmarks to array
            keypoints = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
            
            # Preprocess and predict
            input_tensor = preprocess_frame(keypoints)
            with torch.no_grad():
                output = model(input_tensor)
                probs = F.softmax(output, dim=1)
                pred_idx = torch.argmax(probs, dim=1).item()
                confidence = probs[0, pred_idx].item()
                pred_label = CLASSES[pred_idx]

            # Display prediction on frame
            cv2.putText(frame, f"{pred_label} ({confidence*100:.1f}%)", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("ASL Live Checker", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
