import os
import cv2
import json
import tqdm
import numpy as np
from pathlib import Path


def get_frames(file_path: str):
    vid = cv2.VideoCapture(file_path)

    frames = []

    while(vid.isOpened()):
      # Capture frame-by-frame
        ret, frame = vid.read()
        if ret:
            frames.append(frame)
        else:
            break
    return np.array(frames)


src_dir = Path('Anomaly-Videos-Part-3')
root = Path('data/ucf')
dst_dir = root / 'images'
all_img_dir = Path('data/all_images')

# Create directories
dst_dir.mkdir(exist_ok=True, parents=True)
all_img_dir.mkdir(exist_ok=True, parents=True)

# load all annotations
def load_annotations(annotation_file: str = None) -> dict:
    """Loads the annotations from the specified file."""
    # Default to the annotation file in the UCF annotations directory
    annotation_file = 'data/ucf/annotation_detection/annotations_all.json' if annotation_file is None else annotation_file

    # Load the annotations
    with open(annotation_file, 'r') as f:
        annotations_data = json.load(f)
    
    return annotations_data

# load all frames
def load_image_frames(frame_file: str = None) -> dict:
    """Loads the image frames from the specified file."""
    # Default to the frame file in the UCF data directory
    frame_file = 'data/ucf/frames.json' if frame_file is None else frame_file

    # Load the frame dictionary
    with open(frame_file, 'r') as f:
        frame_dict = json.load(f)

    return frame_dict

# Extract frames from video files
def extract_frames_from_video(frames: dict, dst_path: str = 'data/ucf/images'):
    """Extracts frames from video files and saves them to the specified directory."""
    # Iterate over the video files
    for video_id, val in tqdm.tqdm(frames.items(), desc='Extracting Frames', total=len(frames)):
        # Get the category
        category = ''.join(filter(lambda i: i.isalpha(), video_id))

        # Get the video file path
        video_file_path = f'Anomaly-Videos-Part-3/{category}/{video_id}_x264.mp4'

        # If the video file exists, construct the file path
        if os.path.exists(video_file_path):
            file_path = os.path.join(dst_path, os.path.basename(video_file_path).split('.')[0])
            file_path_all = os.path.join('data/all_images', os.path.basename(video_file_path).split('.')[0])
        
        # if not, skip
        else:
            # print(f'File not found: {video_file_path}')
            continue

        # Load the frames
        frames = get_frames(video_file_path)

        # Iterate over the frames
        for img_idx, frame_idx in val:
            # Get the image
            img = frames[int(frame_idx)]

            # Save the image
            cv2.imwrite(f'{file_path}_{img_idx:04d}.jpg', img)            
            cv2.imwrite(f'{file_path_all}_{img_idx:04d}.jpg', img)


def copy_ucf_images():
    frames = load_image_frames()

    extract_frames_from_video(frames, dst_path='data/ucf/images')


if __name__ == '__main__':
    print()
    # print("Copying UCF images...")
    copy_ucf_images()
    print()