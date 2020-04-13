#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
import argparse, argcomplete
import os, sys
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from numpy_pil_helper import numpy_to_pil_binary

class ImageHelper:
    def __init__(self):
        # Get available method list
        method_list = [func 
                for func in dir(self) 
                if callable(getattr(self, func)) and not func.startswith("__") ]

        parser = argparse.ArgumentParser(
                description='Collection of Image Helper Functions',
                usage='''image_helper.py [-h | --help] <command> [<args>]

List of available image helper functions:
    grayscale_to_binary     Convert grayscale images to binary, p[p > 0] = True
''')
        parser.add_argument(
                'command', choices=method_list, 
                help='The specific image helper function to run')

        argcomplete.autocomplete(parser)
        args = parser.parse_args(sys.argv[1:2])

        if not hasattr(self, args.command):
            print('Unrecognized command "{}"'.format(args.command))
            parser.print_help()
            exit(1)
        getattr(self, args.command)()

    def grayscale_to_binary(self):
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
