import cv2
import os

# Create a folder to save snapshots
os.makedirs("captures", exist_ok=True)
capture_count = 0

# Replace 1 with your OBS Virtual Camera index
cap = cv2.VideoCapture(1)  

if not cap.isOpened():
    print("Cannot open OBS Virtual Camera")
    exit()

print("Press 'c' to capture a frame, 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame = cv2.flip(frame, 1)  # optional: mirror
    cv2.imshow("OBS Virtual Camera Feed", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        # Save a snapshot
        capture_count += 1
        filename = f"captures/frame_{capture_count}.png"
        cv2.imwrite(filename, frame)
        print(f"Saved {filename}")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
