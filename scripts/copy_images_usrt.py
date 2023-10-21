from pathlib import Path
import shutil
import json
import tqdm
from script_utilities import (
    delete_directories,
    create_directories
)


def copy_usrt_images(src_dir: Path, *destinations: tuple[Path]):
    # Load the annotations
    with open("data/usrt/annotation_detection/annotations_all.json", "r") as f:
        ann = json.load(f)
    # Get the list of images
    img_list = set([img['file_name'] for img in ann['images']])
    # Iterate over the source directory
    for path in tqdm.tqdm(src_dir.iterdir(), desc='Copying USRT Images', total=len([*src_dir.iterdir()])):
        # If file is not in the list of images, skip
        if path.name not in img_list:
            continue
        # Copy the image to the destination directories
        for dst_dir in destinations:
            shutil.copyfile(path, dst_dir / path.name)
            
def main():
    # Construct the source directory
    src_dir = Path('Images')
    # Construct the dataset sub-directories
    destinations = [
        Path("data/usrt/images"),
        # Path("data/mgd_usrt"),
        # Path("data/usrt_ucf"),
        Path("data/all_images")
    ]
    # Delete the dataset sub-directories
    delete_directories(*destinations)
    # Create the dataset sub-directories
    create_directories(*destinations)
    # Copy the USRT images
    copy_usrt_images(src_dir, *destinations)



if __name__ == '__main__':
    print()
    main()
    print()