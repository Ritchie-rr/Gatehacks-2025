import os # file/directory operations
import cv2 # OpenCV for reading video files and performing frame operations
import numpy as np # NumPy for handling arrays and saving keypoint sequences
import mediapipe as mp # MediaPipe for hand landmark detection

 
RAW_DATA_DIR = "../data/raw" # Raw video path
KEYPOINT_DIR = "../data/keypoints" # Path for new .npy keypoints
MAX_FRAMES = 60 # Max # of frames per sequence (cut or pad all videos to this length)


mp_hands = mp.solutions.hands # Load the MediaPipe Hands solution

# Create an instance of the Hands model
hands = mp_hands.Hands(
    static_image_mode=False,       # Treat frames as a continuous video stream
    max_num_hands=1,               # Only detect one hand per frame
    min_detection_confidence=0.5,  # Minimum threshold to detect a hand
    min_tracking_confidence=0.5    # Minimum threshold to track landmarks
)

def extract_keypoints_from_video(video_path):
    cap = cv2.VideoCapture(video_path) #Open the video file for reading

    # This will store the sequence of keypoint frames
    sequence = []

    # Read video frame-by-frame
    while True:
        ret, frame = cap.read()     # ret: True/False, frame: image array
        if not ret:                 # Stop if no more frames
            break

        # Convert OpenCV's default BGR format to RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run MediaPipe hand detection on the frame
        results = hands.process(rgb)

        keypoints = []              # Will hold one frame's (x, y, z) coordinates

        # If MediaPipe detected a hand
        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]   # Take the first detected hand

            # Extract all 21 landmarks (each has x, y, z)
            for lm in hand.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])

        else:
            # No hand detected → fill with zeros to keep the frame count consistent
            keypoints = [0.0] * (21 * 3)

        # Add this frame's keypoints to the sequence list
        sequence.append(keypoints)

    # Release video resource
    cap.release()

    # Convert Python list into a NumPy array
    sequence = np.array(sequence)

    # If video is longer than MAX_FRAMES, cut
    if len(sequence) > MAX_FRAMES:
        sequence = sequence[:MAX_FRAMES]

    else:
        # Pad with zeros if video has fewer frames than MAX_FRAMES
        pad_len = MAX_FRAMES - len(sequence)
        padding = np.zeros((pad_len, 21 * 3))
        sequence = np.vstack([sequence, padding])

    return sequence

def process_all_videos():
    # Ensure the output directory exists
    os.makedirs(KEYPOINT_DIR, exist_ok=True)

    # Loop over each gesture class inside raw data directory
    for label in os.listdir(RAW_DATA_DIR):
        raw_label_path = os.path.join(RAW_DATA_DIR, label)

        # Skip anything that isn't a folder (e.g., stray files)
        if not os.path.isdir(raw_label_path):
            continue

        print(f"\n▶ Processing class: {label}")

        # Create output folder for this gesture label
        label_out_dir = os.path.join(KEYPOINT_DIR, label)
        os.makedirs(label_out_dir, exist_ok=True)

        # Loop through all video files inside this class folder
        for video_file in os.listdir(raw_label_path):

            # Only process common video file extensions
            if not video_file.lower().endswith((".mp4")):
                continue

            # Full path to the video file
            video_path = os.path.join(raw_label_path, video_file)
            print(f"   - Extracting: {video_file}")

            # Extract keypoint sequence from this video
            sequence = extract_keypoints_from_video(video_path)

            # Change extension: video.mp4 → video.npy
            out_name = video_file.rsplit(".", 1)[0] + ".npy"
            out_path = os.path.join(label_out_dir, out_name)

            # Save the processed sequence
            np.save(out_path, sequence)

    print("\n✅ Finished extracting all keypoints!")

# When this file is executed directly (not imported), run the pipeline
if __name__ == "__main__":
    process_all_videos()
