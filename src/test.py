import cv2
import os
from datetime import datetime

# Create folders for captures and recordings
os.makedirs("captures", exist_ok=True)
os.makedirs("recordings", exist_ok=True)

capture_count = 0

# Replace 1 with your OBS Virtual Camera index
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Cannot open OBS Virtual Camera")
    exit()

WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
FPS = 20  # Or match your OBS FPS

# Setup video writer (will initialize when recording starts)
out = None
recording = False

print("Press 'c' to capture a frame, 'r' to start/stop recording, 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame = cv2.flip(frame, 1)  # mirror
    cv2.imshow("OBS Virtual Camera Feed", frame)

    key = cv2.waitKey(1) & 0xFF

    # Capture a single frame
    if key == ord('c'):
        capture_count += 1
        capture_filename = f"captures/frame_{capture_count}.png"
        cv2.imwrite(capture_filename, frame)
        print(f"Saved {capture_filename}")

    # Start/stop recording
    elif key == ord('r'):
        recording = not recording
        if recording:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_filename = f"recordings/recording_{timestamp}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_filename, fourcc, FPS, (WIDTH, HEIGHT))
            print("Recording started...")
        else:
            out.release()
            print(f"Recording stopped. Saved to {video_filename}")

    # Quit
    elif key == ord('q'):
        if recording:
            out.release()
            print(f"Recording stopped. Saved to {video_filename}")
        break

    # Write frame to video if recording
    if recording and out is not None:
        out.write(frame)

cap.release()
cv2.destroyAllWindows()
