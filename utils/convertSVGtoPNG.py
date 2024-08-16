import os
from PIL import Image
import cairosvg

def convert_svg_to_png(svg_path, png_path):
    cairosvg.svg2png(url=svg_path, write_to=png_path)

def process_images(directory):
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if file.lower().endswith('.svg'):
                print(file)
                png_path = file_path + '.png'
                convert_svg_to_png(file_path, png_path)
                file_paths.append(png_path)
            else:
                file_paths.append(file_path)
    return file_paths

def open_images(file_paths):
    for file_path in file_paths:
        try:
            img = Image.open(file_path)
        except (PIL.UnidentifiedImageError, IOError):
            print(f"Cannot identify image file {file_path}")

# Replace 'your_directory_path' with the path to your directory
directory_path = '/teamspace/studios/this_studio/diffusion-models/data/pokemon_large/PokemonData'
all_files = process_images(directory_path)

# Open images
images = open_images(all_files)
print('DONE!!')
