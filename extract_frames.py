import cv2
import mediapipe as mp
import os
import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import queue

# Define paths for input videos, output images, and progress tracking file
video_dir = "videos"
output_dir = "images"
progress_file = "progress.json"



CONFIDENT_THRESHOLD = 0.5 # Minimum confidence threshold for face detection
PREVIEW_ENABLED = True # Enable preview of frames on mouse hover
GRID_WIDTH = 1200 # Width of the grid display for frame selection
GRID_HEIGHT = 900 # Height of the grid display for frame selection

MAX_WORKERS = 16  # Number of threads to use
BATCH_SIZE = 100  # Number of videos to process in each batch

# Initialize a thread-safe progress management system
progress_lock = Lock()
if os.path.exists(progress_file):
    with open(progress_file, 'r') as f:
        progress = json.load(f)
else:
    progress = {"completed": [], "frames": {}}

# Setup for MediaPipe's face detection solution
mp_face_detection = mp.solutions.face_detection

# Create directories if they don't exist
os.makedirs(video_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# Function to resize an image while maintaining its aspect ratio
def resize_with_aspect_ratio(image, max_width, max_height):
    h, w = image.shape[:2]
    if w > max_width or h > max_height:
        aspect_ratio = w / h
        if aspect_ratio > 1:
            new_w = max_width
            new_h = int(max_width / aspect_ratio)
        else:
            new_h = max_height
            new_w = int(max_height * aspect_ratio)
        return cv2.resize(image, (new_w, new_h))
    return image

# Function to validate if a frame contains a fully visible face
def is_valid_frame(face_detection_result, frame, frame_width, frame_height):
    if not face_detection_result.detections:
        return False

    detection = face_detection_result.detections[0]
    bbox = detection.location_data.relative_bounding_box
    x, y, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height

    # Ensure the face bounding box is entirely within the frame
    if x < 0 or y < 0 or (x + w) > 1 or (y + h) > 1:
        return False

    return True

# Function to process a single video and extract valid frames
def process_single_video(video_name, model_selection=1, min_detection_confidence=CONFIDENT_THRESHOLD):
    video_path = os.path.join(video_dir, video_name)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Could not open {video_name}. Skipping.")
        return video_name, []

    valid_frames = []
    frame_count = 0

    # Initialize MediaPipe face detection model
    with mp.solutions.face_detection.FaceDetection(model_selection=model_selection, min_detection_confidence=min_detection_confidence) as face_detection:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            # Skip every nth frame for efficiency
            if frame_count % 10 != 0:
                continue

            frame_height, frame_width, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(frame_rgb)

            if is_valid_frame(results, frame, frame_width, frame_height):
                valid_frames.append(frame)

        cap.release()

    # Select frames from the start and end of the video
    if len(valid_frames) > 24:
        valid_frames = valid_frames[:12] + valid_frames[-12:]

    print(f"Valid frames for {video_name} processed.")

    # Update progress if no valid frames are found
    if not valid_frames:
        print(f"No valid frames found for {video_name}.")
        progress["completed"].append(video_name)
        with open(progress_file, 'w') as f:
            json.dump(progress, f, indent=4)

    return video_name, valid_frames

# Function to display frames and allow user selection
def display_and_select_frames(video_name, valid_frames):
    if not valid_frames:
        print(f"No valid frames to display for {video_name}.")
        progress["completed"].append(video_name)
        with open(progress_file, 'w') as f:
            json.dump(progress, f, indent=4)
        return

    selected_frames = {"before": None, "after": None}
    w, h = valid_frames[0].shape[:2]
    grid_img = np.zeros((w * 2, h * 12, 3), dtype=np.uint8)

    # Arrange frames in a grid
    for i, frame in enumerate(valid_frames):
        row = i // 12
        col = i % 12
        grid_img[row * w:(row + 1) * w, col * h:(col + 1) * h] = frame

    grid_img = resize_with_aspect_ratio(grid_img, GRID_WIDTH, GRID_HEIGHT)

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            col = int(x / (grid_img.shape[1] / 12))
            row = int(y / (grid_img.shape[0] / 2))
            idx = row * 12 + col
            if idx < len(valid_frames):
                if selected_frames["before"] is None:
                    selected_frames["before"] = valid_frames[idx]
                    print(f"Selected frame {idx} as 'before'.")
                elif selected_frames["after"] is None:
                    selected_frames["after"] = valid_frames[idx]
                    print(f"Selected frame {idx} as 'after'.")
                else:
                    print("Both frames selected. Press 'q' to exit.")
        # Preview the frame under the mouse cursor if preview is enabled
        if event == cv2.EVENT_MOUSEMOVE and PREVIEW_ENABLED:
            col = int(x / (grid_img.shape[1] / 12))
            row = int(y / (grid_img.shape[0] / 2))
            idx = row * 12 + col
            if idx < len(valid_frames):
            # Display the preview of the selected frame
                cv2.imshow(f"Preview", resize_with_aspect_ratio(valid_frames[idx], GRID_HEIGHT, GRID_HEIGHT))


    cv2.imshow(f"Select Frames for {video_name}", grid_img)
    cv2.setMouseCallback(f"Select Frames for {video_name}", click_event)

    while selected_frames["before"] is None or selected_frames["after"] is None:
        if cv2.waitKey(1) == ord('q'):
            print("Selection interrupted. Exiting.")
            progress["completed"].append(video_name)
            with open(progress_file, 'w') as f:
                json.dump(progress, f, indent=4)
            break

    cv2.destroyAllWindows()

    # Save selected frames
    os.makedirs(output_dir, exist_ok=True)
    if selected_frames["before"] is not None:
        cv2.imwrite(os.path.join(output_dir, f"{video_name}_before.jpg"), selected_frames["before"])
    if selected_frames["after"] is not None:
        cv2.imwrite(os.path.join(output_dir, f"{video_name}_after.jpg"), selected_frames["after"])

    progress["completed"].append(video_name)
    with open(progress_file, 'w') as f:
        json.dump(progress, f, indent=4)

# Main program logic
video_queue = queue.Queue()

# Populate the video queue with pending videos
for video_name in os.listdir(video_dir):
    if video_name not in progress.get('completed', []):
        if video_name + "_before.jpg" not in os.listdir(output_dir):
            if video_name + "_after.jpg" not in os.listdir(output_dir):
                video_queue.put(video_name)

# Start multi-threaded processing in batches
result_queue = queue.Queue()


while not video_queue.empty():
    batch_futures = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for _ in range(min(BATCH_SIZE, video_queue.qsize())):
            video_name = video_queue.get()
            future = executor.submit(process_single_video, video_name)
            batch_futures.append(future)

        for future in as_completed(batch_futures):
            video_name, valid_frames = future.result()
            if valid_frames:
                result_queue.put((video_name, valid_frames))

    # Display frames after each batch
    while not result_queue.empty():
        video_name, valid_frames = result_queue.get()
        display_and_select_frames(video_name, valid_frames)

print("Processing completed.")
