from pathlib import Path
import matplotlib.pyplot as plt
import PIL 
import shutil
import json
import tqdm

src_dir = Path("MGD/MGD2020/JPEGImages/")
dst_dir = Path("data/mgd/images")
all_img_dir = Path("data/all_images")

shutil.rmtree(dst_dir, ignore_errors=True)

dst_dir.mkdir(exist_ok=True, parents=True)
all_img_dir.mkdir(exist_ok=True, parents=True)

def copy_mgd_images():
    with open("data/mgd/annotation_detection/annotations_all.json", "r") as f:
        ann = json.load(f)

    img_list = set([img['file_name'] for img in ann['images']])
    total_file_names = len(tuple(src_dir.iterdir()))

    for path in tqdm.tqdm(src_dir.iterdir(), desc='Copying MGD Images', total=total_file_names):
        if path.name not in img_list:
            continue
        try:
            plt.imread(path)
            shutil.copyfile(path, dst_dir / path.name)
            shutil.copyfile(path, all_img_dir / path.name)

        except PIL.UnidentifiedImageError:
            # Some images in the original source have a byte error

            with path.open("rb") as f:
                temp = f.read()
            for dest_path in [dst_dir / path.name, all_img_dir / path.name]:
                with dest_path.open("wb") as f:
                    f.write(temp.lstrip(b"\x00"))


if __name__ == '__main__':
    print()
    # print("Copying MGD images...")
    copy_mgd_images()
    print()