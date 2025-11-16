import os # file/directory operations
import cv2 # OpenCV for reading video files and performing frame operations
import numpy as np # NumPy for handling arrays and saving keypoint sequences
import mediapipe as mp # MediaPipe for hand landmark detection

 
RAW_DATA_DIR = "../data/raw" # Raw video path
KEYPOINT_DIR = "../data/keypoints" # Path for new .npy keypoints

LABEL_DIR = "../data/labels"  # Path for text label files

MAX_FRAMES = 60 # Max number of frames per sequence (cut or pad all videos to this length)


mp_hands = mp.solutions.hands # Load the MediaPipe Hands solution

# Total dims = 126 (hands)
FEATURE_DIM = 126


# Create an instance of the Hands model
hands = mp_hands.Hands(
    static_image_mode=False,       # Treat frames as a continuous video stream
    max_num_hands=2,               # Only detect one hand per frame
    min_detection_confidence=0.5,  # Minimum threshold to detect a hand
    min_tracking_confidence=0.5    # Minimum threshold to track landmarks
)


def extract_keypoints_from_video(video_path):
    cap = cv2.VideoCapture(video_path) #Open the video file for reading

    sequence = []
    last_valid = np.zeros(FEATURE_DIM)  # start with zeros

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_results = hands.process(rgb)

        keypoints = []

        # ----------------------------------------------------
        # Hand Landmarks (21 points × 3 x 2(hands) = 126 dims)
        # ----------------------------------------------------
        if hand_results.multi_hand_landmarks:
            hand = hand_results.multi_hand_landmarks[0]

            for lm in hand.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])

            keypoints = np.array(keypoints)
            last_valid = keypoints.copy()  # update last valid
        else:
            # No hand → reuse last valid frame
            keypoints = last_valid.copy()

        sequence.append(keypoints)

    cap.release()
    sequence = np.array(sequence)

    num_frames = sequence.shape[0]

    if num_frames >= MAX_FRAMES:
        sequence = sequence[:MAX_FRAMES]
    else:
        pad_len = MAX_FRAMES - num_frames
        # pad using last frame, NOT zeros
        padding = np.tile(sequence[-1], (pad_len, 1))
        sequence = np.vstack([sequence, padding])

    return sequence

def process_all_videos():
    # Ensure output directories exist
    os.makedirs(KEYPOINT_DIR, exist_ok=True)

    # Ensure label directory exists
    os.makedirs(LABEL_DIR, exist_ok=True)

    # Global counter for flat numbering
    counter = 1

    # Loop over each gesture class inside raw data directory
    for label in os.listdir(RAW_DATA_DIR):
        raw_label_path = os.path.join(RAW_DATA_DIR, label)

        # Skip anything that isn't a folder (e.g., stray files)
        if not os.path.isdir(raw_label_path):
            continue

        print(f"\n▶ Processing class: {label}")

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

            # Use sequential global filenames (video1, video2...)
            base_name = f"video{counter}"

            # Save to flat keypoints directory
            out_path = os.path.join(KEYPOINT_DIR, base_name + ".npy")

            # Save the processed sequence
            np.save(out_path, sequence)

            # Create matching label text file
            label_path = os.path.join(LABEL_DIR, base_name + ".txt")
            with open(label_path, "w") as f:
                f.write(label)  # the folder name (e.g., "hello")

            # Increment counter
            counter += 1

    print("\n✅ Finished extracting all keypoints and label files!")


# When this file is executed directly (not imported), run the pipeline
if __name__ == "__main__":
    process_all_videos()