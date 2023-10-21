from pathlib import Path
import shutil
import json
import tqdm

src_dir = Path("Images")
dst_dir = Path("data/usrt/images")
all_img_dir = Path("data/all_images")

shutil.rmtree(dst_dir, ignore_errors=True)

dst_dir.mkdir(exist_ok=True, parents=True)
all_img_dir.mkdir(exist_ok=True, parents=True)

def copy_usrt_images():
    with open("data/usrt/annotation_detection/annotations_all.json", "r") as f:
        ann = json.load(f)

    img_list = set([img['file_name'] for img in ann['images']])

    for path in tqdm.tqdm(src_dir.iterdir(), desc='Copying USRT Images', total=len(tuple(src_dir.iterdir()))):
        if path.name not in img_list:
            continue
        shutil.copyfile(path, dst_dir / path.name)
        shutil.copyfile(path, all_img_dir / path.name)


if __name__ == '__main__':
    print()
    # print("Copying USRT images...")
    copy_usrt_images()
    print()