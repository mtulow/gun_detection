import shutil
from pathlib import Path


def delete_directories(*subdirs: tuple[Path], all_directories: bool = False):
    """Delete the given dataset sub-directories."""
    # Determine if the directory containing all images should be deleted
    subdirs = subdirs if all_directories else subdirs[:-1]
    # Delete the sub-directories
    for subdir in subdirs:
        shutil.rmtree(subdir, ignore_errors=True)
    return

def create_directories(*subdirs: tuple[Path]):
    """Create the given dataset sub-directories."""
    for subdir in subdirs:
        subdir.mkdir(exist_ok=True, parents=True)
    return
