from pathlib import Path
import matplotlib.pyplot as plt
import PIL 
import shutil
import json
import tqdm
from scripts.script_utilities import delete_directories, create_directories



def copy_mgd_images(src_dir: Path, *destinations: tuple[Path]):
    # Load the annotations
    with open("data/mgd/annotation_detection/annotations_all.json", "r") as f:
        ann = json.load(f)
    # Get the list of images
    img_list = set([img['file_name'] for img in ann['images']])
    # Iterate over the source directory
    for path in tqdm.tqdm(src_dir.iterdir(), desc='Copying MGD Images', total=len([*src_dir.iterdir()])):
        if path.name not in img_list:
            continue
        try:
            plt.imread(path)
            for dst_dir in destinations:
                shutil.copyfile(path, dst_dir / path.name)

        except PIL.UnidentifiedImageError:
            # Some images in the original source have a byte error
            with path.open("rb") as f:
                temp = f.read()
            for dst_dir in destinations:
                with (dst_dir / path.name).open("wb") as f:
                    f.write(temp.lstrip(b"\x00"))

def main():
    # Construct the source directory
    src_dir = Path('MGD/MGD2020/JPEGImages')
    # Construct the dataset sub-directories
    destinations = [
        Path("data/mgd/images"),
        # Path("data/dataset_pairs/mgd_usrt"),
        # Path("data/dataset_pairs/ucf_mgd"),
        Path("data/all_images"),
    ]
    # Delete the dataset sub-directories
    delete_directories(*destinations)
    # Create the dataset sub-directories
    create_directories(*destinations)
    # Copy the MGD images
    copy_mgd_images(src_dir, *destinations)
    print('Done!')


if __name__ == '__main__':
    print()
    main()
    print()