#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
"""Post Processing Script"""
import os
import glob
import gc
import argparse
import argcomplete

from PIL import Image
import numpy as np
from scipy import ndimage
from skimage import morphology
from tqdm import tqdm

from networks.dataset import save_predicted_masks
from utils.compute_mask_accuracy import ComputeMaskAccuracy

Image.MAX_IMAGE_PIXELS = None

def get_parser() -> argparse.ArgumentParser:
    """Get the argparse parser for this script"""
    main_parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Post processing\n\t')

    post_proc_parser = main_parser.add_argument_group('Post processing configurations')
    post_proc_parser.add_argument(
        "mask_dir", type=str, 
        help="Directory of predicted mask files (e.g. data/outputs/UNet)")
    post_proc_parser.add_argument(
        "--truth-dirs", type=str, nargs='+',
        default=['/BrainSeg/data/box_Ab/groundtruth',
                 '/BrainSeg/data/box_control/groundtruth'],
        help="Directories of groundtruth .png files")
    post_proc_parser.add_argument(
        "--compute-accuracy", action='store_true',
        help="Compute accuracy after post processing the masks (Default: False)")

    return main_parser

def post_proc(args) -> None:
    """Start post processing based on args input"""
    # Convert to abspath
    args.mask_dir = os.path.abspath(args.mask_dir)
    # Glob mask .png filepaths
    mask_paths = sorted([p for p in glob.glob(os.path.join(args.mask_dir, "*.png")) 
                         if 'Gray' not in p and 'White' not in p and 'Back' not in p])
    print(f'\n\tFound {len(mask_paths)} masks in {args.mask_dir}')

    # Create output directory
    save_dir = args.mask_dir + '_postproc'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print(f'\tSaving processed mask to "{save_dir}"')

    success_count = 0
    for mask_path in tqdm(mask_paths):
        svs_name = mask_path.split('/')[-1].replace('.png', '')

        mask_arr = np.array(Image.open(mask_path))
        gray_mask_arr = (mask_arr == 1)
        white_mask_arr = (mask_arr == 1)
        del mask_arr

        # Apply morphological opening on GM
        gray_mask_arr = ndimage.binary_opening(gray_mask_arr, morphology.disk(radius=8))

        # Apply morphological opening on WM
        white_mask_arr = ndimage.binary_opening(white_mask_arr, morphology.disk(radius=8))

        # Reconstruct mask_arr
        mask_arr = np.zeros_like(gray_mask_arr, dtype='uint8')
        mask_arr[gray_mask_arr] = 1
        mask_arr[white_mask_arr] = 2
        del gray_mask_arr, white_mask_arr

        save_predicted_masks(mask_arr, save_dir, svs_name)
        del mask_arr

        success_count += 1
        gc.collect()

    # Print summary
    print('\nOut of %d WSIs, \n\t%d were successfully processed'
          % (len(mask_paths), success_count))

    # If run compute_mask_accuracy.py at the end
    if args.compute_accuracy:
        compute_acc_parser = ComputeMaskAccuracy.get_parser()
        args = compute_acc_parser.parse_args([save_dir] + args.truth_dirs)
        ComputeMaskAccuracy(args)
    else:
        print(f'To compute mask accuracy, please run compute_mask_accuracy.py {save_dir}')

if __name__ == '__main__':
    parser = get_parser()
    argcomplete.autocomplete(parser)
    post_proc_args = parser.parse_args()

    post_proc(post_proc_args)
