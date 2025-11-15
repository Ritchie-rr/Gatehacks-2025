import cv2
import pyvirtualcam

cap = cv2.VideoCapture(0)

with pyvirtualcam.Camera(width=1920, height=1080, fps=30) as cam:
    print(f'Using virtual camera: {cam.device}')
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        cam.send(frame)
        cam.sleep_until_next_frame()
