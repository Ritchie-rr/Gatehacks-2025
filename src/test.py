# Filename: obs_virtual_cam_demo.py
import cv2
import pyvirtualcam

# --- CONFIGURATION ---
# Match these to your OBS Virtual Camera settings
WIDTH = 640
HEIGHT = 480
FPS = 20

# Open your default webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

# Start the virtual camera (OBS should already be running with Virtual Camera started)
with pyvirtualcam.Camera(width=WIDTH, height=HEIGHT, fps=FPS) as cam:
    print(f'Virtual camera started: {cam.device}')
    print("Press 'q' in the preview window to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from webcam")
            break

        # Flip frame horizontally (mirror)
        frame = cv2.flip(frame, 1)

        # Optional: overlay text for testing
        cv2.putText(frame, "ASL Demo", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2, cv2.LINE_AA)

        # Send the frame to the virtual camera
        cam.send(frame)
        cam.sleep_until_next_frame()

        # Show a local preview window
        cv2.imshow("Virtual Cam Preview", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting...")
            break

# Release resources
cap.release()
cv2.des
