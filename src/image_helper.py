#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
"""Image Helper Script"""
import os
import sys
import time
import glob
import argparse
import argcomplete

from tqdm import tqdm
import numpy as np
from PIL import Image

from utils.numpy_pil_helper import numpy_to_pil_binary, numpy_to_pil_palette
Image.MAX_IMAGE_PIXELS = None

def grayscale_to_binary(args: argparse.Namespace) -> None:
    """Convert grayscale images to binary and overwrites"""
    # Resolve path from os.getcwd()
    args.input_dir = os.path.abspath(args.input_dir)

    # Get all PNG images in input_dir
    img_paths = sorted(glob.glob(os.path.join(args.input_dir, "*.png")))

    if not img_paths:
        print(f'No .png images found in "{args.input_dir}"')
        sys.exit(0)

    success_count, error_count = 0, 0
    for img_path in tqdm(img_paths):
        # Read in png image
        img_arr = np.array(Image.open(img_path))

        # Image format checks
        if img_arr.dtype == np.bool:
            print('\t[%5s] This image is already in binary format. '
                  'Skipping it.' % ("INFO"))
            continue
        if not img_arr.dtype == np.uint8:
            print('\t[%5s] Wrong numpy dtype, '
                  'expected "np.uint8" but got "%s". Skipping it.'
                  % ("ERROR", img_arr.dtype))
            error_count += 1
            continue
        if not img_arr.ndim == 2:
            # pylint: disable=bad-string-format-type
            print('\t[%5s] Wrong numpy dimension, '
                  'expected "2" but got "%d" with shape %s. Skipping it.'
                  % ("ERROR", img_arr.ndim, img_arr.shape))
            # pylint: enable=bad-string-format-type
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
          % (len(img_paths), success_count, error_count))

def get_thumbnails(args: argparse.Namespace) -> None:
    """Save thumbnails of the images"""
    # Resolve path from os.getcwd()
    args.input_dir = os.path.abspath(args.input_dir)

    # Get all PNG images in input_dir
    img_paths = sorted(glob.glob(os.path.join(args.input_dir, "*.png")))

    if not img_paths:
        print(f'No .png images found in "{args.input_dir}"')
        sys.exit(0)

    # Maximum size
    max_size = (args.max_width, args.max_height)

    # Output directory is a sub-directory in args.input_dir named thumbnails
    output_dir = os.path.join(args.input_dir, "thumbnails")

    # Make output directory if not exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f'Found {len(img_paths)} images. Saving to {output_dir}')
    for img_path in tqdm(img_paths):
        img_name = img_path.split('/')[-1]
        img_save_path = os.path.join(output_dir, img_name)

        # Read in png image
        img = Image.open(img_path)

        # Convert to a thumbnail and save
        img.thumbnail(max_size)
        img.save(img_save_path)

    print('Done!')

def combine_truth_binary(args: argparse.Namespace) -> None:
    """Combine groundtruth binary images into palette images"""
    # Resolve path from os.getcwd()
    args.input_dir = os.path.abspath(args.input_dir)

    # Get image_names in input_dir
    img_names = [pathname.split('/')[-1].replace('-Gray.png', '') for pathname in
                 glob.glob(os.path.join(args.input_dir, "*-Gray.png"))]
    img_names = sorted(img_names)

    print(f'Found {len(img_names)} images. '
          f'Saving to same folder as {img_names[0]}.png')
    for img_name in tqdm(img_names):
        truth_back_path = os.path.join(args.input_dir, img_name+'-Background.png')
        truth_gray_path = os.path.join(args.input_dir, img_name+'-Gray.png')
        truth_white_path = os.path.join(args.input_dir, img_name+'-White.png')
        save_mask_path = os.path.join(args.input_dir, img_name+'.png')

        truth_back_arr = np.array(Image.open(truth_back_path))
        truth_gray_arr = np.array(Image.open(truth_gray_path))
        truth_white_arr = np.array(Image.open(truth_white_path))

        mask_arr = np.zeros_like(truth_back_arr, dtype='uint8')
        mask_arr[truth_gray_arr] = 1
        mask_arr[truth_white_arr] = 2

        mask_img = numpy_to_pil_palette(mask_arr)
        # For label 0, leave as black color
        # For label 1, set to cyan color: R0G255B255
        # For label 2, set to yellow color: R255G255B0
        mask_img.putpalette([0, 0, 0, 0, 255, 255, 255, 255, 0])
        mask_img.save(save_mask_path)
        del mask_arr, mask_img

    print('Done!')

class SubcommandHelpFormatter(argparse.RawDescriptionHelpFormatter):
    """For removing the extra line below subcommands in argparse"""
    def _format_action(self, action):
        parts = super(SubcommandHelpFormatter, self)._format_action(action)
        if action.nargs == argparse.PARSER:
            parts = "\n".join(parts.split("\n")[1:])
        return parts

def get_parser():
    """Get the argparse parser for this script"""
    main_parser = argparse.ArgumentParser(
        formatter_class=SubcommandHelpFormatter,
        description='Collection of Image Helper Functions\n\t'
                    '\xa9 2020 Runlin Guo (https://github.com/KolinGuo)\n\t'
                    f'Last updated: {time.ctime(os.path.getmtime(__file__))}')
    subparsers = main_parser.add_subparsers(title='subcommands',
                                            dest='subcommand')

    p_thumbnail = subparsers.add_parser(
        'get_thumbnails',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Save thumbnails of the images',
        help='Save thumbnails of the images')
    p_thumbnail.add_argument(
        'input_dir', type=str,
        help='Input directory of PNG images (e.g. "./data/outputs")')
    p_thumbnail.add_argument(
        '--max-width', type=int, default=2000,
        help='Thumbnail max width in pixels (Default: 2000)')
    p_thumbnail.add_argument(
        '--max-height', type=int, default=2000,
        help='Thumbnail max height in pixels (Default: 2000)')
    p_thumbnail.set_defaults(func=get_thumbnails)

    p_gray2binary = subparsers.add_parser(
        'gray_to_binary',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Convert grayscale images to binary and overwrites',
        help='Convert grayscale images to binary, p[p > 0] = True')
    p_gray2binary.add_argument(
        'input_dir', type=str,
        help='Input directory of PNG grayscale images (e.g. "./data/outputs")')
    p_gray2binary.set_defaults(func=grayscale_to_binary)

    p_combinebinary = subparsers.add_parser(
        'combine_truth_binary',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Combine groundtruth binary images into palette images',
        help='Combine groundtruth binary images into palette images')
    p_combinebinary.add_argument(
        'input_dir', type=str,
        help='Input directory of PNG binary groundtruth images (e.g. "./groundtruth")')
    p_combinebinary.set_defaults(func=combine_truth_binary)

    return main_parser

if __name__ == '__main__':
    parser = get_parser()
    argcomplete.autocomplete(parser)
    helper_args = parser.parse_args()

    if helper_args.subcommand is None:
        parser.print_help()
        sys.exit(0)

    helper_args.func(helper_args)
