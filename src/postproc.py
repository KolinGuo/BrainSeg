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

def method_1(mask_arr: "NDArray[np.uint8]") -> "NDArray[np.uint8]":
    """Binary_opening for GM and WM"""
    gray_mask_arr = (mask_arr == 1)
    white_mask_arr = (mask_arr == 2)
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
    return mask_arr

def method_2(mask_arr: "NDArray[np.uint8]") -> "NDArray[np.uint8]":
    """Area_opening followed by area_closing (Remove local maxima and minima)"""

    # Apply area_opening to remove local maxima with area < 200000 px
    mask_arr = morphology.area_opening(mask_arr, area_threshold=200000)
    print('Finish area_opening')

    # Apply area_closing to remove local minima with area < 200000 px
    mask_arr = morphology.area_closing(mask_arr, area_threshold=200000)
    print('Finish area_closing')

    return mask_arr

def method_3(mask_img: "Image", down_factor=4) -> "NDArray[np.uint8]":
    """Downsample =>
    Area_opening followed by area_closing (Remove local maxima and minima) =>
    Upsample"""

    width, height = mask_img.width, mask_img.height

    # Downsample the image
    mask_arr = np.array(
        mask_img.resize((width // down_factor, height // down_factor), Image.NEAREST))
    del mask_img
    print('Finish downsampling')

    # Apply area_opening to remove local maxima with area < 200000 px
    mask_arr = morphology.area_opening(mask_arr, area_threshold=320000 // down_factor**2)
    print('Finish area_opening')

    # Apply area_closing to remove local minima with area < 200000 px
    mask_arr = morphology.area_closing(mask_arr, area_threshold=320000 // down_factor**2)
    print('Finish area_closing')

    # Upsample the output
    mask_arr = np.array(Image.fromarray(mask_arr).resize((width, height), Image.NEAREST))
    print('Finish upsampling')

    return mask_arr

def method_4(mask_img: "Image", down_factor=4) -> "NDArray[np.uint8]":
    """Downsample => Area_opening (Remove local maxima) =>
    Swap index of GM and WM => Area_opening => Swap index back =>
    Upsample"""
    # pylint: disable=invalid-name
    def swap_GM_WM(arr):
        """Swap GM and WM in arr (swaps index 1 and index 2)"""
        arr_1 = (arr == 1)
        arr[arr == 2] = 1
        arr[arr_1] = 2
        del arr_1
        return arr
    # pylint: enable=invalid-name

    width, height = mask_img.width, mask_img.height

    # Downsample the image
    mask_arr = np.array(
        mask_img.resize((width // down_factor, height // down_factor), Image.NEAREST))
    del mask_img
    print('Finish downsampling')

    # Apply area_opening to remove local maxima with area < 200000 px
    mask_arr = morphology.area_opening(mask_arr, area_threshold=320000 // down_factor**2)
    print('Finish area_opening #1')

    # Swap index of GM and WM
    mask_arr = swap_GM_WM(mask_arr)
    print('Finish swapping index')

    # Apply area_opening to remove local maxima with area < 200000 px
    mask_arr = morphology.area_opening(mask_arr, area_threshold=320000 // down_factor**2)
    print('Finish area_opening #2')

    # Swap index back
    mask_arr = swap_GM_WM(mask_arr)
    print('Finish swapping index back')

    # Upsample the output
    mask_arr = np.array(Image.fromarray(mask_arr).resize((width, height), Image.NEAREST))
    print('Finish upsampling')

    return mask_arr

def method_5(mask_img: "Image", down_factor=4) -> "NDArray[np.uint8]":
    """Downsample => Area_opening (Remove local maxima) =>
    Swap index of GM and WM => Area_opening => Swap index back =>
    Morphological opening => Upsample"""
    # pylint: disable=invalid-name
    def swap_GM_WM(arr):
        """Swap GM and WM in arr (swaps index 1 and index 2)"""
        arr_1 = (arr == 1)
        arr[arr == 2] = 1
        arr[arr_1] = 2
        del arr_1
        return arr
    # pylint: enable=invalid-name

    width, height = mask_img.width, mask_img.height

    # Downsample the image
    mask_arr = np.array(
        mask_img.resize((width // down_factor, height // down_factor), Image.NEAREST))
    del mask_img
    print('Finish downsampling')

    # Apply area_opening to remove local maxima with area < 200000 px
    mask_arr = morphology.area_opening(mask_arr, area_threshold=320000 // down_factor**2)
    print('Finish area_opening #1')

    # Swap index of GM and WM
    mask_arr = swap_GM_WM(mask_arr)
    print('Finish swapping index')

    # Apply area_opening to remove local maxima with area < 200000 px
    mask_arr = morphology.area_opening(mask_arr, area_threshold=320000 // down_factor**2)
    print('Finish area_opening #2')

    # Swap index back
    mask_arr = swap_GM_WM(mask_arr)
    print('Finish swapping index back')

    # Apply opening with disk-shaped kernel (r=8) to smooth boundary
    mask_arr = morphology.opening(mask_arr, selem=morphology.disk(radius=32 // down_factor))
    print('Finish morphological opening')

    # Upsample the output
    mask_arr = np.array(Image.fromarray(mask_arr).resize((width, height), Image.NEAREST))
    print('Finish upsampling')

    return mask_arr

def method_6(mask_img: "Image", down_factor=4) -> "NDArray[np.uint8]":
    """Downsample => Area_opening (Remove local maxima) =>
    Swap index of GM and WM => Area_opening => Swap index back =>
    Area_closing => Morphological opening => Upsample"""
    # pylint: disable=invalid-name
    def swap_GM_WM(arr):
        """Swap GM and WM in arr (swaps index 1 and index 2)"""
        arr_1 = (arr == 1)
        arr[arr == 2] = 1
        arr[arr_1] = 2
        del arr_1
        return arr
    # pylint: enable=invalid-name

    width, height = mask_img.width, mask_img.height

    # Downsample the image
    mask_arr = np.array(
        mask_img.resize((width // down_factor, height // down_factor), Image.NEAREST))
    del mask_img
    print('Finish downsampling')

    # Apply area_opening to remove local maxima with area < 20000 px
    mask_arr = morphology.area_opening(mask_arr, area_threshold=320000 // down_factor**2)
    print('Finish area_opening #1')

    # Swap index of GM and WM
    mask_arr = swap_GM_WM(mask_arr)
    print('Finish swapping index')

    # Apply area_opening to remove local maxima with area < 20000 px
    mask_arr = morphology.area_opening(mask_arr, area_threshold=320000 // down_factor**2)
    print('Finish area_opening #2')

    # Swap index back
    mask_arr = swap_GM_WM(mask_arr)
    print('Finish swapping index back')

    # Apply area_closing to remove local minima with area < 12500 px
    mask_arr = morphology.area_closing(mask_arr, area_threshold=200000 // down_factor**2)
    print('Finish area_closing')

    # Apply opening with disk-shaped kernel (r=8) to smooth boundary
    mask_arr = morphology.opening(mask_arr, selem=morphology.disk(radius=32 // down_factor))
    print('Finish morphological opening')

    # Upsample the output
    mask_arr = np.array(Image.fromarray(mask_arr).resize((width, height), Image.NEAREST))
    print('Finish upsampling')

    return mask_arr

def get_boundary(mask_arr: "NDArray[np.uint8]") -> "NDArray[np.uint8]":
    """Extract GM/WM boundary"""
    gray_mask_arr = (mask_arr == 1)
    white_mask_arr = (mask_arr == 2)
    del mask_arr

    # Extract GM boundary
    gray_mask_arr ^= ndimage.binary_erosion(gray_mask_arr, morphology.square(width=3))
    print('Finish GM boundary')

    # Extract WM boundary
    white_mask_arr ^= ndimage.binary_erosion(white_mask_arr, morphology.square(width=3))
    print('Finish WM boundary')

    # Reconstruct mask_arr
    mask_arr = np.zeros_like(gray_mask_arr, dtype='uint8')
    mask_arr[gray_mask_arr] = 1
    mask_arr[white_mask_arr] = 2
    del gray_mask_arr, white_mask_arr
    return mask_arr

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

        ##### Method 1 #####
        #mask_arr = np.array(Image.open(mask_path))
        #mask_arr = method_1(mask_arr)

        ##### Method 2 #####
        #mask_arr = np.array(Image.open(mask_path))
        #mask_arr = method_2(mask_arr)

        ##### Method 3 #####
        #mask_img = Image.open(mask_path)
        #mask_arr = method_3(mask_img)
        #del mask_img

        ##### Method 4 #####
        #mask_img = Image.open(mask_path)
        #mask_arr = method_4(mask_img)
        #del mask_img

        ##### Method 5 #####
        #mask_img = Image.open(mask_path)
        #mask_arr = method_5(mask_img)
        #del mask_img

        ##### Method 6 ##### TODO
        #mask_img = Image.open(mask_path)
        #mask_arr = method_6(mask_img)
        #del mask_img

        ##### Get boundary #####
        mask_arr = np.array(Image.open(mask_path))
        mask_arr = get_boundary(mask_arr)

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
