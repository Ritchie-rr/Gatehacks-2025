import os # file/directory operations
import cv2 # OpenCV for reading video files and performing frame operations
import numpy as np # NumPy for handling arrays and saving keypoint sequences
import mediapipe as mp # MediaPipe for hand landmark detection

 
RAW_DATA_DIR = "../data/raw" # Raw video path
KEYPOINT_DIR = "../data/keypoints" # Path for new .npy keypoints

LABEL_DIR = "../data/labels"  # Path for text label files

MAX_FRAMES = 60 # Max number of frames per sequence (cut or pad all videos to this length)


mp_hands = mp.solutions.hands # Load the MediaPipe Hands solution
mp_face = mp.solutions.face_mesh # Load the MediaPipe Face solution

# Face and Head Landmark index list
FACE_IDX = {
    "right_eye":  [33, 133, 160, 159, 158, 157, 173, 144],
    "left_eye":   [362, 263, 387, 386, 385, 384, 398, 373],
    "right_brow": [46, 53, 52, 65, 55],
    "left_brow":  [276, 283, 282, 295, 285],
    "mouth_outer":[61,146,91,181,84,17,314,405,321,375],
    "mouth_inner":[78,95,88,178,87,14,317,402,310,415]
}

# Combined 46 face points → 46 × 3 = 138 dims
SELECTED_FACE_LANDMARKS = (
    FACE_IDX["right_eye"] +
    FACE_IDX["left_eye"] +
    FACE_IDX["right_brow"] +
    FACE_IDX["left_brow"] +
    FACE_IDX["mouth_outer"] +
    FACE_IDX["mouth_inner"]
)

# 7 head movement points → 7 × 3 = 21 dims
HEAD_IDX = [
    1,    # nose tip
    6,    # nose bridge
    152,  # chin
    234,  # left temple
    454,  # right temple
    33,   # left eye outer
    263   # right eye outer
]

# Total dims = 63 (hands) + 138 (face) + 21 (head) = 222
FEATURE_DIM = 222


# Create an instance of the Hands model
hands = mp_hands.Hands(
    static_image_mode=False,       # Treat frames as a continuous video stream
    max_num_hands=2,               # Only detect one hand per frame
    min_detection_confidence=0.5,  # Minimum threshold to detect a hand
    min_tracking_confidence=0.5    # Minimum threshold to track landmarks
)

# Create an instance of the Face model
face = mp_face.FaceMesh(
    static_image_mode=False, # Same comments as above
    max_num_faces=1, 
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
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

        # Run both hand and face models
        hand_results = hands.process(rgb)
        face_results = face.process(rgb)

        # Will hold one frame's (x, y, z) coordinates
        keypoints = []              

        # ----------------------------------------------------
        # Hand Landmarks (21 points × 3 = 63 dims)
        # ----------------------------------------------------
        if hand_results.multi_hand_landmarks:
            hand = hand_results.multi_hand_landmarks[0]   # first detected hand

            for lm in hand.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])
        else:
            keypoints.extend([0.0] * 63) # Needs to be 63
    
        # ----------------------------------------------------
        # Face Landmarks (46 points × 3 = 138 dims)
        # ----------------------------------------------------
        if face_results.multi_face_landmarks:
            face_lm = face_results.multi_face_landmarks[0]

            for idx in SELECTED_FACE_LANDMARKS:
                lm = face_lm.landmark[idx]
                keypoints.extend([lm.x, lm.y, lm.z])
        else:
            keypoints.extend([0.0] * 138) # Needs to be 138-dim

        # ----------------------------------------------------
        # Head Movement Landmarks (7 points × 3 = 21 dims)
        # ----------------------------------------------------
        if face_results.multi_face_landmarks:
            face_lm = face_results.multi_face_landmarks[0]  # (uses same face result)

            for idx in HEAD_IDX:
                lm = face_lm.landmark[idx]
                keypoints.extend([lm.x, lm.y, lm.z])
        else:
            keypoints.extend([0.0] * 21) # Needs to be 21-dim

        # Append 222-dim frame vector
        sequence.append(keypoints)

    # Release video resource
    cap.release()

    # Convert Python list into a NumPy array
    sequence = np.array(sequence)

    ### >>> FIXED: Pad or crop based on 222 feature dims
    num_frames = sequence.shape[0]

    if num_frames >= MAX_FRAMES:
        sequence = sequence[:MAX_FRAMES]
    else:
        pad_len = MAX_FRAMES - num_frames
        padding = np.zeros((pad_len, FEATURE_DIM))
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
