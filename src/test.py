import pyvirtualcam
import cv2

cap = cv2.VideoCapture(0)

with pyvirtualcam.Camera(width=640, height=480, fps=20, backend='obs') as cam:
    print(f'Using virtual camera: {cam.device}')
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        cam.send(frame)
        cam.sleep_until_next_frame()