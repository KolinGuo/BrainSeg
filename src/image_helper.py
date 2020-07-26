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
import pyvips as Vips

from utils.numpy_pil_helper import numpy_to_pil_binary, numpy_to_pil_palette
from networks.dataset import save_predicted_masks
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

def resize_to_original(args: argparse.Namespace) -> None:
    """Resize the image to original size"""
    # Resolve path from os.getcwd()
    args.input_dir = os.path.abspath(args.input_dir)
    args.ref_dir = [os.path.abspath(p) for p in args.ref_dir]

    # Get all img_paths in ref_dir
    svs_paths = sorted([p for d in args.ref_dir
                        for p in glob.glob(os.path.join(d, '*AB*.svs'))])

    # Output directory is input_dir appended with _resized
    output_dir = args.input_dir + "_resized"

    # Make output directory if not exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f'Found {len(svs_paths)} reference images. Saving to {output_dir}')

    for svs_path in tqdm(svs_paths):
        svs_name = svs_path.split('/')[-1].replace('.svs', '')

        # Get the mask image in input_dir whose name matches svs_name
        img_path = glob.glob(os.path.join(args.input_dir, svs_name+'.png'))[0]

        # Get reference width and height
        vips_img = Vips.Image.new_from_file(svs_path, level=0)
        width, height = vips_img.width, vips_img.height

        # Open the mask image
        img = Image.open(img_path)

        # If image is in RGB mode, convert it to P mode
        if img.mode == 'RGB':
            img_arr = np.array(img)

            # For label 0, leave as black color
            new_img_arr = np.zeros(img_arr.shape[:-1], dtype=np.uint8)
            # For label 1, set to cyan color: R0G255B255
            new_img_arr = np.where(np.all(img_arr == (0, 255, 255), axis=-1), 1, new_img_arr)
            # For label 2, set to yellow color: R255G255B0
            new_img_arr = np.where(np.all(img_arr == (255, 255, 0), axis=-1), 2, new_img_arr)

            # Tests results using assert
            np.testing.assert_array_equal(
                np.all(img_arr == (0, 0, 0), axis=-1), new_img_arr == 0)
            np.testing.assert_array_equal(
                np.all(img_arr == (0, 255, 255), axis=-1), new_img_arr == 1)
            np.testing.assert_array_equal(
                np.all(img_arr == (255, 255, 0), axis=-1), new_img_arr == 2)

            img = numpy_to_pil_palette(new_img_arr)
            img.putpalette([0, 0, 0, 0, 255, 255, 255, 255, 0])

        print(f"{svs_name}: ({img.width}, {img.height}) => ({width}, {height})")
        assert img.mode == 'P', f"Wrong PIL image mode, expected P get {img.mode}"

        # Resize the image
        img = img.resize((width, height))
        img_arr = np.array(img)

        save_predicted_masks(img_arr, output_dir, svs_name)

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

    p_resizeoriginal = subparsers.add_parser(
        'resize_to_original',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""Resize the image to a reference size.
        Takes an input_dir and a ref_dir as inputs.
        Resize images in input_dir to match the size of those SVS images in ref_dir.
        Images are matched by their names.""",
        help='Resize the image to a reference size')
    p_resizeoriginal.add_argument(
        'input_dir', type=str,
        help='Input directory of PNG images (e.g. "./data/outputs")')
    p_resizeoriginal.add_argument(
        'ref_dir', type=str, nargs='+',
        help='Reference directories of SVS images(e.g. "./data/box_Ab ./data/box_control")')
    p_resizeoriginal.set_defaults(func=resize_to_original)

    return main_parser

if __name__ == '__main__':
    parser = get_parser()
    argcomplete.autocomplete(parser)
    helper_args = parser.parse_args()

    if helper_args.subcommand is None:
        parser.print_help()
        sys.exit(0)

    helper_args.func(helper_args)
