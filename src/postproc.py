#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
"""Post Processing Script"""
# pylint: disable=invalid-name

import os
import glob
import gc
import argparse
import argcomplete

from PIL import Image
import numpy as np
from scipy import ndimage
from skimage import morphology, measure
import lxml.etree as ET
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
        "--method-num", type=int, choices=range(1, 7), default=6,
        help="Method used for post-processing (Default: method_6)")
    post_proc_parser.add_argument(
        "--truth-dirs", type=str, nargs='+',
        default=['/BrainSeg/data/box_Ab/groundtruth',
                 '/BrainSeg/data/box_control/groundtruth'],
        help="Directories of groundtruth .png files")
    post_proc_parser.add_argument(
        "--xml-downsample-rate", type=int, default=50,
        help="Downsample rate for xml annotation (Default: 50)")
    post_proc_parser.add_argument(
        "--only-convert-xml", action='store_true',
        help="Only convert input masks to ImageScope xml annotations (Default: False)")
    post_proc_parser.add_argument(
        "--extract-boundary", action='store_true',
        help="Extract boundary after post processing the masks (Default: False)")
    post_proc_parser.add_argument(
        "--only-extract-boundary", action='store_true',
        help="Only extract boundary of input masks (Default: False)")
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
    area_threshold_prop = 0.05
    area_threshold = int(area_threshold_prop * width * height // down_factor**2)

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

    # Apply remove_small_objects to remove tissue residue with area < 0.05 * width * height
    tissue_arr = morphology.remove_small_objects(mask_arr > 0, min_size=area_threshold,
                                                 connectivity=2)
    mask_arr[np.invert(tissue_arr)] = 0
    del tissue_arr
    print('Finish remove_small_objects')

    # Apply opening with disk-shaped kernel (r=8) to smooth boundary
    mask_arr = morphology.opening(mask_arr, selem=morphology.disk(radius=32 // down_factor))
    print('Finish morphological opening')

    # Upsample the output
    mask_arr = np.array(Image.fromarray(mask_arr).resize((width, height), Image.NEAREST))
    print('Finish upsampling')

    return mask_arr

def extract_boundary(mask_arr: "NDArray[np.uint8]") -> "NDArray[np.uint8]":
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

def convert_mask_to_xml(mask_arr: "NDArray[np.uint8]", xml_downsample_rate: int) \
        -> "ET.ElementTree":
    """Convert mask to ImageScope xml annotation"""
    def get_regions_node(parent_node: "ET.Element") -> "ET.Element":
        """Get a Regions node under parent_node"""
        regions_node = ET.SubElement(parent_node, 'Regions')
        headers_node = ET.SubElement(regions_node, 'RegionAttributeHeaders')
        ET.SubElement(headers_node, 'AttributeHeader',
                      attrib={'Id': "9999", 'Name': "Region", 'ColumnWidth': "-1"})
        ET.SubElement(headers_node, 'AttributeHeader',
                      attrib={'Id': "9997", 'Name': "Length", 'ColumnWidth': "-1"})
        ET.SubElement(headers_node, 'AttributeHeader',
                      attrib={'Id': "9996", 'Name': "Area", 'ColumnWidth': "-1"})
        ET.SubElement(headers_node, 'AttributeHeader',
                      attrib={'Id': "9998", 'Name': "Text", 'ColumnWidth': "-1"})
        return regions_node

    def build_boundary_xml(binary_mask_arr: "NDArray[bool]", regions_node: "ET.Element",
                           downsample_rate: int) -> "ET.Element":
        """Build the Vertices of contours of binary_mask_arr"""
        contours = measure.find_contours(binary_mask_arr, 0, fully_connected='high')

        for i, contour in enumerate(contours):
            region_node = ET.SubElement(regions_node, 'Region',
                                        attrib={'Id': str(i+1), 'DisplayId': str(i+1)})
            vertices_node = ET.SubElement(region_node, 'Vertices')
            for y, x in contour[::downsample_rate]:
                ET.SubElement(vertices_node, 'Vertex',
                              attrib={'X': str(int(x)), 'Y': str(int(y)), 'Z': "0"})
        return regions_node

    # Create base structure
    # ET.SubElement(parent, tag, attrib={}, **extra)
    root = ET.Element('Annotations')
    GM_node = ET.SubElement(root, 'Annotation', attrib={'Id': "1", 'Name': "Gray Matter"})
    WM_node = ET.SubElement(root, 'Annotation', attrib={'Id': "2", 'Name': "White Matter"})

    GM_regions_node = get_regions_node(GM_node)
    WM_regions_node = get_regions_node(WM_node)

    _ = ET.SubElement(GM_node, 'Plots')
    _ = ET.SubElement(WM_node, 'Plots')

    # Build boundaries
    GM_regions_node = build_boundary_xml(mask_arr == 1, GM_regions_node, xml_downsample_rate)
    WM_regions_node = build_boundary_xml(mask_arr == 2, WM_regions_node, xml_downsample_rate)

    return ET.ElementTree(root)

def apply_post_proc_method(mask_path: str, method_num: int) -> "NDArray[np.uint8]":
    """Apply post proceeing method"""
    if method_num == 1:
        ##### Method 1 #####
        mask_arr = np.array(Image.open(mask_path))
        mask_arr = method_1(mask_arr)
    elif method_num == 2:
        ##### Method 2 #####
        mask_arr = np.array(Image.open(mask_path))
        mask_arr = method_2(mask_arr)
    elif method_num == 3:
        ##### Method 3 #####
        mask_img = Image.open(mask_path)
        mask_arr = method_3(mask_img)
        del mask_img
    elif method_num == 4:
        ##### Method 4 #####
        mask_img = Image.open(mask_path)
        mask_arr = method_4(mask_img)
        del mask_img
    elif method_num == 5:
        ##### Method 5 #####
        mask_img = Image.open(mask_path)
        mask_arr = method_5(mask_img)
        del mask_img
    elif method_num == 6:
        ##### Method 6 #####
        mask_img = Image.open(mask_path)
        mask_arr = method_6(mask_img)
        del mask_img
    else:
        raise ValueError('Unknown method number')
    return mask_arr

def post_proc(args) -> None:
    """Start post processing based on args input"""
    # Convert to abspath
    args.mask_dir = os.path.abspath(args.mask_dir)
    # Glob mask .png filepaths
    mask_paths = sorted([p for p in glob.glob(os.path.join(args.mask_dir, "*.png"))
                         if 'Gray' not in p and 'White' not in p and 'Back' not in p])
    print(f'\n\tFound {len(mask_paths)} masks in {args.mask_dir}')

    # If not post-processing
    post_processing = False
    if args.only_convert_xml or args.only_extract_boundary:
        save_dir = args.mask_dir    # Save in input directory
    else:
        post_processing = True
        # Create output directory
        save_dir = args.mask_dir + '_postproc'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    print(f'\tSaving processed mask to "{save_dir}"')

    success_count = 0
    for mask_path in tqdm(mask_paths):
        svs_name = mask_path.split('/')[-1].replace('.png', '')

        ##### Apply post-processing method #####
        if post_processing:
            mask_arr = apply_post_proc_method(mask_path, args.method_num)
            # Save post-processed masks
            save_predicted_masks(mask_arr, save_dir, svs_name)
        else:   # Do not apply post-processing
            mask_arr = np.array(Image.open(mask_path))

        ##### Convert to XML #####
        xml_tree = convert_mask_to_xml(mask_arr, args.xml_downsample_rate)
        save_xml_dir = os.path.join(save_dir, 'xml_annotations')
        if not os.path.exists(save_xml_dir):
            os.makedirs(save_xml_dir)
        save_xml_path = os.path.join(save_xml_dir, svs_name+'.xml')
        # Save xml annotations
        xml_tree.write(save_xml_path, pretty_print=True)
        del xml_tree

        ##### Get boundary #####
        if args.extract_boundary or args.only_extract_boundary:
            mask_arr = extract_boundary(mask_arr)
            save_boundary_dir = os.path.join(save_dir, 'boundary')
            if not os.path.exists(save_boundary_dir):
                os.makedirs(save_boundary_dir)
            # Save boundary masks
            save_predicted_masks(mask_arr, save_boundary_dir, svs_name)
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
# pylint: enable=invalid-name
