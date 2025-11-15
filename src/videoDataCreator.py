import cv2
import os
from datetime import datetime

# Ask for the gesture label
gesture_name = input("Enter the gesture name: ").strip()

# Create folders for this gesture
record_dir = os.path.join("recordings", gesture_name)
os.makedirs(record_dir, exist_ok=True)

capture_count = 0

# Replace 1 with your OBS Virtual Camera index
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Cannot open OBS Virtual Camera")
    exit()

WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
FPS = 20  # Or match your OBS FPS

# Setup video writer (initialized when recording starts)
out = None
recording = False

print("\nControls:")
print("  'c' → capture a single frame")
print("  'r' → start/stop recording video")
print("  'q' → quit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame = cv2.flip(frame, 1)  # mirror
    cv2.imshow(f"{gesture_name} - OBS Feed", frame)

    key = cv2.waitKey(1) & 0xFF

    # Capture a single frame
    if key == ord('c'):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        capture_filename = os.path.join(capture_dir, f"frame_{timestamp}.png")
        cv2.imwrite(capture_filename, frame)
        print(f"Saved snapshot: {capture_filename}")

    # Start/stop recording
    elif key == ord('r'):
        recording = not recording
        if recording:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_filename = os.path.join(record_dir, f"recording_{timestamp}.mp4")
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
