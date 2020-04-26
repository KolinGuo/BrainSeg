#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
import os, glob, time
import gc   # Garbage Collector interface
import argparse, argcomplete
import numpy as np
import re
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

def img_frombytes(data):
    size = data.shape[::-1]
    databytes = np.packbits(data, axis=1)
    return Image.frombytes(mode='1', size=size, data=databytes)

def combine_annotation_tiles(args):
    args.tile_dir = os.path.abspath(args.tile_dir)

    SAVE_DIR = os.path.abspath(os.path.join(args.tile_dir, '../combined_tiles'))

    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    image_names = [n.split('/')[-1].split('(')[0][:-1]
            for n in glob.glob(os.path.join(args.tile_dir, '*.txt'))]

    for i, imagename in enumerate(image_names):
        print('[%3d/%3d]\tProcessing annotation tiles of %s' 
                % (i+1, len(image_names), imagename))

        key_back = 0
        key_w = 1
        key_g = 2
    
        # Read tile keys
        with open(glob.glob(os.path.join(args.tile_dir, imagename + '*key.txt'))[0], 'r') as f:
            for line in f:
                elements = re.split('\t|\n', line)
                if elements[0] == "White Matter":
                    key_w = int(elements[1])
                elif elements[0] == "Grey Matter":
                    key_g = int(elements[1])

        # Get all tiles for this image
        tilenames = glob.glob(os.path.join(args.tile_dir, imagename + '*.png'))
    
        # Save for tile dimensions
        xywh = np.zeros((len(tilenames), 4), dtype='int')
    
        # Read in dimensions
        for j, tilename in enumerate(tilenames):
            xywh[j, :] = re.split('\(|,|\)', tilename)[2:-1]
        
        # Sort xywh
        xywh = xywh[np.lexsort((xywh.T[1], xywh.T[0]))]
    
        # Calculate width and height
        width = xywh[-1, 0] + xywh[-1, 2]
        height = xywh[-1, 1] + xywh[-1, 3]
    
        # Create save images
        save_img_back = Image.new('1', (width, height))
        save_img_grey = Image.new('1', (width, height))
        save_img_white = Image.new('1', (width, height))
    
        for j, tilename in enumerate(tilenames):
            # Read x, y start
            start_x, start_y = re.split('\(|,|\)', tilename)[2:4]
            start_x, start_y = int(start_x), int(start_y)
    
            # Read in tile image
            tile_arr = np.array(Image.open(tilename))
            # Create grey white arrays
            back_arr = np.zeros_like(tile_arr, dtype='bool')
            grey_arr = np.zeros_like(tile_arr, dtype='bool')
            white_arr = np.zeros_like(tile_arr, dtype='bool')
    
            # Set background to 1 for back_arr
            back_arr[tile_arr == key_back] = True
            # Set grey matter to 1 for grey_arr
            grey_arr[tile_arr == key_g] = True
            # Set white matter to 1 for white_arr
            white_arr[tile_arr == key_w] = True

            # Paste the images
            save_img_back.paste(img_frombytes(back_arr), (start_x, start_y))
            save_img_grey.paste(img_frombytes(grey_arr), (start_x, start_y))
            save_img_white.paste(img_frombytes(white_arr), (start_x, start_y))
            del tile_arr, back_arr, grey_arr, white_arr
    
        # Save the images
        save_img_back.save(os.path.join(SAVE_DIR, imagename + '-Background.png'))
        save_img_grey.save(os.path.join(SAVE_DIR, imagename + '-Grey.png'))
        save_img_white.save(os.path.join(SAVE_DIR, imagename + '-White.png'))
        del save_img_back, save_img_grey, save_img_white
        gc.collect()

def get_parser():
    parser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description='Combine Annotation Tiles\n\t'
            '\xa9 2020 Runlin Guo (https://github.com/KolinGuo)\n\t'
            f'Last updated: {time.ctime(os.path.getmtime(__file__))}')

    parser.add_argument('tile_dir', type=str, 
            help='Path to "exported_tiles" folder (e.g. "./exported_tiles")')

    return parser

if __name__ == '__main__':
    # Parse command-line arguments
    parser = get_parser()
    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    combine_annotation_tiles(args)

    print("Done!")
