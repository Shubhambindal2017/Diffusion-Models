import os
import PIL
from PIL import Image

def process_images(directory):
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
    return file_paths

def open_images(file_paths):
    for file_path in file_paths:
        try:
            img = Image.open(file_path)
            sizeSet.add(img.size)
        except (PIL.UnidentifiedImageError, IOError):
            print(f"Cannot identify image file {file_path}")

# Replace 'your_directory_path' with the path to your directory
directory_path = '/teamspace/studios/this_studio/diffusion-models/data/animefacedataset'
all_files = process_images(directory_path)

sizeSet = set()
# Open images
images = open_images(all_files)
print('DONE!!')
print(len(sizeSet))
print(sorted(sizeSet))