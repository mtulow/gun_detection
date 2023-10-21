import os
import cv2
import json
import numpy as np
from pathlib import Path
from script_utilities import delete_directories, create_directories


def get_frames(file_path: Path) -> np.ndarray:
    """Returns the frames from the specified video file."""
    # Open the video file
    vid = cv2.VideoCapture(file_path)
    # Check if the video file was opened successfully
    if not vid.isOpened():
        raise ValueError(f'Unable to open video file: {file_path}')
    # Initialize the frames list
    frames = []
    # Iterate over the frames
    while(vid.isOpened()):
        # Capture frame-by-frame
        ret, frame = vid.read()
        if ret:
            frames.append(frame)
        else:
            break
    return np.array(frames)

# load all annotations
def load_annotations(annotation_file: str = None) -> dict:
    """Loads the annotations from the specified file."""
    # Default to the annotation file in the UCF annotations directory
    annotation_file = 'data/ucf/annotation_detection/annotations_all.json' \
        if annotation_file is None \
            else annotation_file
    # Load the annotations
    with open(annotation_file, 'r') as f:
        annotations_data = json.load(f)
    
    return annotations_data

# load all frames
def load_image_frames(frame_file: str = None) -> dict:
    """Loads the image frames from the specified file."""
    # Default to the frame file in the UCF data directory
    frame_file = 'data/ucf/frames.json' \
        if frame_file is None \
            else frame_file
    # Load the frame dictionary
    with open(frame_file, 'r') as f:
        frame_dict = json.load(f)
    return frame_dict

# Extract frames from video files
def extract_frames_from_video(frames: dict, src_dir: Path, *destinations: tuple[Path]):
    """Extracts frames from video files and saves them to the specified directory."""
    # Iterate over the video files
    for video_id, val in frames.items():
        # Construct the video file path
        video_file_path = str(src_dir \
            / ''.join(filter(lambda i: i.isalpha(), video_id)) \
            / f'{video_id}_x264.mp4')

        # If the video file doesn't exists, skip
        if not os.path.exists(video_file_path):
            continue

        # Load the frames
        frames = get_frames(video_file_path)

        # Iterate over the frames
        for img_idx, frame_idx in val:
            # Get the image
            img = frames[int(frame_idx)]

            # Save image to destination directories
            for dst_dir in destinations:
                # Save the image
                cv2.imwrite(f'{dst_dir}/{img_idx:04d}.jpg', img)


def copy_ucf_images():
    # Construct source directory
    src_dir = Path('Anomaly-Videos-Part-3')
    # Construct dataset sub-directories
    destinations = [
        Path('data/ucf/images'),
        # Path('data/dataset_pairs/usrt_ucf'),
        # Path('data/dataset_pairs/ucf_mgd'),
        Path('data/all_images'),
    ]
    # Delete the destination sub-directories
    delete_directories(*destinations)
    # Create the destination sub-directories
    create_directories(*destinations)
    # Load frame dictionary for dataset images
    frames = load_image_frames()
    # Extract frames from mp4 file into dataset sub-directory
    extract_frames_from_video(frames, src_dir, *destinations)


if __name__ == '__main__':
    print()
    print('Copying UCF images...')
    copy_ucf_images()
    print('Done!')
    print()
