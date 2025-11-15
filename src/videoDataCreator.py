import cv2
import os

# Ask for the gesture name
gesture_name = input("Enter the gesture name: ").strip()

# Ask for starting number
start_num = int(input("What number are you on? ").strip())
current_num = start_num

# Create folders for this gesture
record_dir = os.path.join("recordings", gesture_name)
os.makedirs(record_dir, exist_ok=True)

# Replace 1 with your OBS Virtual Camera index
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Cannot open OBS Virtual Camera")
    exit()

WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
FPS = 30 

# Setup video writer (initialized when recording starts)
out = None
recording = False

print("\nControls:")
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

    # Start/stop recording
    if key == ord('r'):
        recording = not recording
        if recording:
            video_filename = os.path.join(record_dir, f"{gesture_name}_{current_num}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_filename, fourcc, FPS, (WIDTH, HEIGHT))
            print(f"Recording started: {video_filename}")
        else:
            out.release()
            print(f"Recording stopped: {video_filename}")
            current_num += 1  # increment for next recording

    # Quit
    elif key == ord('q'):
        if recording:
            out.release()
            print(f"Recording stopped: {video_filename}")
        break

    # Write frame to video if recording
    if recording and out is not None:
        out.write(frame)

cap.release()
cv2.destroyAllWindows()
