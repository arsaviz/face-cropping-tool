import cv2
import mediapipe as mp
import os

CONFIDENT_THRESHOLD = 0.5 # Minimum confidence threshold for face detection

def crop_face(image_path, min_detection_confidence=CONFIDENT_THRESHOLD):
    """
    Detects and crops the face from an input image using MediaPipe Face Detection.

    Args:
        image_path (str): Path to the input image.

    Returns:
        numpy.ndarray: Cropped face image, or None if no face is detected.
    """
    # Initialize MediaPipe Face Detection
    mp_face_detection = mp.solutions.face_detection

    # Load the input image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Failed to load image: {image_path}")
        return None

    height, width, _ = image.shape

    # Convert the image to RGB (required by MediaPipe)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform face detection
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=min_detection_confidence) as face_detection:
        results = face_detection.process(image_rgb)

        # Check if any face is detected
        if not results.detections:
            print(f"No faces detected in {image_path}.")
            return None

        # Extract bounding box information for the first detected face
        detection = results.detections[0]
        bboxC = detection.location_data.relative_bounding_box
        x_min = int(bboxC.xmin * width)
        y_min = int(bboxC.ymin * height)
        bbox_width = int(bboxC.width * width)
        bbox_height = int(bboxC.height * height)

        # Ensure the bounding box coordinates are within image dimensions
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        bbox_width = min(width - x_min, bbox_width)
        bbox_height = min(height - y_min, bbox_height)

        # Crop the face from the image
        cropped_face = image[y_min:y_min + bbox_height, x_min:x_min + bbox_width]

        return cropped_face

def crop_faces_in_directory(input_dir, output_dir):
    """
    Processes all images in the specified input directory, detects and crops faces, 
    and saves the cropped images to the output directory.

    Args:
        input_dir (str): Path to the directory containing input images.
        output_dir (str): Path to the directory where cropped images will be saved.
    """
    # Create the output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Output directory created: {output_dir}")

    # List all files in the input directory
    images = os.listdir(input_dir)

    # Process each image with a progress bar
    for image in images:
        input_path = os.path.join(input_dir, image)
        output_path = os.path.join(output_dir, image)

        try:
            # Detect and crop face
            cropped_face = crop_face(input_path)
            if cropped_face is not None:
                # Save the cropped face to the output directory
                cv2.imwrite(output_path, cropped_face)
                print(f"Face cropped and saved: {output_path}")
            else:
                print(f"No face detected in: {input_path}")
        except Exception as e:
            print(f"Error processing {input_path}: {e}")

if __name__ == "__main__":
    """
    Main script to detect and crop faces from images in a specified directory.
    """
    # Define input and output directories
    input_directory = 'images'  # Directory containing the input images
    output_directory = 'images_cropped'  # Directory to save cropped images

    # Start the face cropping process
    crop_faces_in_directory(input_directory, output_directory)
