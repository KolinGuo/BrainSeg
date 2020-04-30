import os, glob
import numpy as np
import pyvips as Vips
from tqdm import tqdm
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
PIL_TILE_SIZE = 30000
from typing import Any
from nptyping import NDArray

from numpy_pil_helper import numpy_to_pil_rgb

import logging
logger = logging.getLogger('svs_to_png')

# map vips formats to np dtypes
format_to_dtype = {
    'uchar': np.uint8,
    'char': np.int8,
    'ushort': np.uint16,
    'short': np.int16,
    'uint': np.uint32,
    'int': np.int32,
    'float': np.float32,
    'double': np.float64,
    'complex': np.complex64,
    'dpcomplex': np.complex128,
}

def vips2numpy(vi) -> NDArray[Any]:
    '''
    vips image to numpy array.
    
    Input:
        vi : a pyvips Image object.
    Output:
        out : a numpy ndarray.
    '''
    return np.ndarray(buffer=vi.write_to_memory(),
                      dtype=format_to_dtype[vi.format],
                      shape=(vi.height, vi.width, vi.bands))

def svs_to_numpy(svs_file_path: str, 
        save_png: bool=False, save_png_dir: str=None) -> NDArray[np.uint8]:
    '''
    Convert a svs file to a numpy array.
    
    Inputs:
        svs_file_path : A String to a .svs WSI file.
        save_png      : boolean, True for saving a png file, default is False.
        save_png_dir  : A String to saving directory path, default is None.
    Output:
        out : a numpy array representing an RGB image, (..., ..., 3).
    '''
    assert os.path.isfile(svs_file_path), 'Input .svs file "{}" does not exist!'.format(svs_file_path)

    vips_img = Vips.Image.new_from_file(svs_file_path, level=0)
    width, height = vips_img.width, vips_img.height

    img_arr = vips2numpy(vips_img)
    img_arr = img_arr[..., 0:3]     # remove alpha channel

    if save_png and save_png_dir is not None:
        if not os.path.exists(save_png_dir):
            os.makedirs(save_png_dir)
        img_name = svs_file_path.split('/')[-1].replace('.svs', '.png')

        # Convert numpy array to PIL image
        orig_img = numpy_to_pil_rgb(img_arr)
        save_path = os.path.join(save_png_dir, img_name)
        orig_img.save(save_path)
        logger.info('Saved orig_img as %s', save_path)
        del orig_img

    return img_arr.astype('uint8')

def svs_to_png_batch(input_dir: str, output_dir: str) -> None:
    '''
    Convert all svs files in input_dir to png images saved in output_dir.
    
    Inputs:
        input_dir  : A String to input directory containing svs files.
        output_dir : A String to output directory for saving converted png files.
    '''
    assert os.path.isdir(input_dir), 'Input directory "{}" does not exist!'.format(input_dir)

    # Make output directory if not exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    img_names = [pathname.split('/')[-1] for pathname in glob.glob(os.path.join(input_dir, "*.svs"))]
    img_names = sorted(img_names)

    t = tqdm(total=len(img_names), postfix='\n', leave=False)
    for i, img_name in enumerate(img_names):
        vips_img = Vips.Image.new_from_file(os.path.join(input_dir, img_name), level=0)
        width, height = vips_img.width, vips_img.height

        # tdqm progress bar
        t.set_description_str("Image " + img_name + ', ' + '({}, {})'.format(width, height), refresh=False)

        # Iteration of creating PIL image tile by tile
        iters = np.ceil([height / PIL_TILE_SIZE, width / PIL_TILE_SIZE]).astype('int')

        orig_img = Image.new('RGB', (width, height))
        for row in range(iters[0]):
            for col in range(iters[1]):
                # Get start and end pixel location
                start_r, start_c = row * PIL_TILE_SIZE, col * PIL_TILE_SIZE
                end_r, end_c = min(height, start_r + PIL_TILE_SIZE), min(width, start_c + PIL_TILE_SIZE)

                # Paste the tile into image
                orig_tile = vips2numpy(vips_img.crop(start_c, start_r, end_c-start_c, end_r-start_r))
                orig_tile = orig_tile[..., 0:3]     # remove alpha channel
                orig_tile = Image.fromarray(orig_tile, 'RGB')
                orig_img.paste(orig_tile, (start_c, start_r))

        orig_img.save(os.path.join(output_dir, img_name.replace('.svs', '.png')))

        t.update()
    t.close()

if __name__ == '__main__':
    ### For testing ###
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("input_dir", type=str, help="Input Directory of .svs files")
    parser.add_argument("output_dir", type=str, help="Output Directory of .png files")
    args = parser.parse_args()

    print("Input directory: " + args.input_dir)
    print("Output directory: " + args.output_dir)

    svs_to_png_batch(args.input_dir, args.output_dir)
    print("Done!")
