#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
import argparse, argcomplete
import os, sys, glob
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from numpy_pil_helper import numpy_to_pil_binary

class ImageHelper:
    def __init__(self) -> None:
        # Get available method list
        method_list = [func 
                for func in dir(self) 
                if callable(getattr(self, func)) and not func.startswith("__") ]

        parser = argparse.ArgumentParser(
                description='Collection of Image Helper Functions',
                usage='''image_helper.py [-h | --help] <command> [<args>]

List of available image helper functions:
    grayscale_to_binary     Convert grayscale images to binary, p[p > 0] = True
    get_thumbnails          Save a thumbnail of the images, MAXSIZE=(2000,2000)
''')
        parser.add_argument(
                'command', choices=method_list, 
                help='The specific image helper function to run')

        argcomplete.autocomplete(parser)
        args = parser.parse_args(sys.argv[1:2])

        getattr(self, args.command)()

    def get_thumbnails(self) -> None:
        parser = argparse.ArgumentParser(
                description='Save a thumbnail of the images',
                usage='''image_helper.py get_thumbnails <input_dir>\n''')
        parser.add_argument('input_dir', type=str, 
                help='Input directory of images')
        args = parser.parse_args(sys.argv[2:3])

        # Resolve path from os.getcwd()
        args.input_dir = os.path.abspath(args.input_dir)

        # Get all PNG images in input_dir
        img_paths = [pathname for pathname in 
                glob.glob(os.path.join(args.input_dir, "*.png"))]
        img_paths = sorted(img_paths)

        # Maximum size
        max_size = (2000, 2000)

        output_dir = os.path.join(args.input_dir, "thumbnails")
        
        # Make output directory if not exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for i, img_path in enumerate(img_paths):
            img_name = img_path.split('/')[-1]
            img_save_path = os.path.join(output_dir, img_name)

            print("Image {}/{} {}"\
                    .format(i+1, len(img_paths), img_name))

            # Read in png image
            img = Image.open(img_path)

            # Convert to a thumbnail and save
            img.thumbnail(max_size)
            img.save(img_save_path)

            print('\t[%5s] Done! Saves to %s' % ("INFO", img_save_path))

    def grayscale_to_binary(self) -> None:
        parser = argparse.ArgumentParser(
                description='Convert grayscale images to binary and overwrites',
                usage='''image_helper.py grayscale_to_binary <pathspec>\n''')
        parser.add_argument('pathspec', type=str, nargs='*', 
                help='Input grayscale images')
        args = parser.parse_args(sys.argv[2:])

        # Resolve path from os.getcwd()
        args.pathspec = [os.path.abspath(s) for s in args.pathspec]

        success_count, error_count = 0, 0
        for i, img_path in enumerate(args.pathspec):
            print("Image {}/{} {}"\
                    .format(i+1, len(args.pathspec), img_path.split('/')[-1]))

            # Read in png image
            img_arr = np.array(Image.open(img_path))

            # Image format checks
            if img_arr.dtype == np.bool:
                print('\t[%5s] This image is already in binary format. '
                        'Skipping it.' % ("INFO"))
                continue
            elif not img_arr.dtype == np.uint8:
                print('\t[%5s] Wrong numpy dtype, '
                        'expected "np.uint8" but got "%s". Skipping it.' 
                        % ("ERROR", img_arr.dtype))
                error_count += 1
                continue
            elif not img_arr.ndim == 2:
                print('\t[%5s] Wrong numpy dimension, '
                        'expected "2" but got "%d" with shape %s. Skipping it.' 
                        % ("ERROR", img_arr.ndim, img_arr.shape))
                error_count += 1
                continue

            # Convert to binary
            img_arr[img_arr > 0] = 1
            img_arr = img_arr.astype('bool')

            # Convert to PIL image and overwrites
            img = numpy_to_pil_binary(img_arr)
            img.save(img_path)

            print('\t[%5s] Done! Overwrites to %s' % ("INFO", img_path))
            success_count += 1

        print('\nOut of %d images, \n\t%d were successfully converted'
                '\n\t%d have encountered an error\n'
                % (len(args.pathspec), success_count, error_count))

if __name__ == '__main__':
    ImageHelper()
