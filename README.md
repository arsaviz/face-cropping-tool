# Video Frame Extraction and Face Cropping

This repository contains two Python scripts for face detection and cropping:

1. `extract_frames.py`: Processes video files to detect and select frames with visible faces.
2. `crop_faces.py`: Detects and crops faces from images within a specified directory

## Features

- **Video Frame Selection**: Detects faces in videos, allows frame selection, and saves selected frames.
- **Resumable Processing**: Tracks progress to allow resuming processing from where it left off.
- **Face Detection**: Uses MediaPipe for accurate face detection.
- **Image Cropping**: Automatically crops detected faces and saves them in an output directory.

## Requirements

- Python 3.7 or higher
- OpenCV for image processing
- MediaPipe for face detection
  
Refer to the [requirements.txt](requirements.txt) file for detailed dependencies.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/arsaviz/face-cropping-tool.git
   cd face-cropping-tool
   ```

2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Prepare input data:

   - Place videos in the `videos/` directory for `extract_frames.py`.
   - Place images in the `images/` directory for `crop_faces.py`. (Alternatively, run `extract_frames.py` to generate the `images/` directory automatically.)
  
## Configuration

These constants can be adjusted directly in the script files before execution to customize their behavior:

- Face Detection Threshold:
  - `CONFIDENT_THRESHOLD = 0.5`: Minimum confidence for face detection.
- Preview Grid Dimensions (`extract_frames.py`):
  - `GRID_WIDTH = 1200`, `GRID_HEIGHT = 900`: Adjust grid display size for frame selection.
- Batch Processing (`extract_frames.py`):
  - `MAX_WORKERS = 16`, `BATCH_SIZE = 100`: Configure thread usage and batch size.

## Usage

### 1. Video Face Frame Selection

Run the `extract_frames.py` script to process videos:

   ```bash
   python extract_frames.py
   ```

- The script detects valid frames containing visible faces.
- Frames are displayed in a grid for manual selection
  - click grid images for selection of before and after images in order (if no valid frame exsists press q to skip video)
- Selected frames are saved in the `images/` directory.

### 2. Image Face Cropping

Run the `crop_faces.py` script to crop faces from images:

   ```bash
   python crop_faces.py
   ```

The cropped images will be saved in the `images_cropped/` directory.

## Notes

- Ensure that your input images/videos are placed in the correct directories before running the scripts.
- The `extract_frames.py` script is designed to efficiently handle large datasets with integrated progress tracking.

## License

This project is open-source and licensed under the MIT License.

## Contributing

Feel free to submit issues or contribute improvements to the scripts by opening a pull request.
