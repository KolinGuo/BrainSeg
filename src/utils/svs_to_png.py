import os, glob
import numpy as np
import pyvips as Vips
from tqdm import tqdm
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
PIL_TILE_SIZE = 30000

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

# vips image to numpy array
def vips2numpy(vi):
    return np.ndarray(buffer=vi.write_to_memory(),
                      dtype=format_to_dtype[vi.format],
                      shape=[vi.height, vi.width, vi.bands])

def svs_to_png(input_dir, output_dir):
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
        iters = np.uint8(np.ceil([height / PIL_TILE_SIZE, width / PIL_TILE_SIZE]))

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
    args = argparse.ArgumentParser()

    args.add_argument("input_dir", type=str, help="Input Directory of .svs files")
    args.add_argument("output_dir", type=str, help="Output Directory of .png files")
    args = args.parse_args()

    print("Input directory: " + args.input_dir)
    print("Output directory: " + args.output_dir)

    svs_to_png(args.input_dir, args.output_dir)
    print("Done!")
