import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
PIL_TILE_SIZE = 30000
from nptyping import NDArray

def numpy_to_pil_binary(img_arr: NDArray[bool]):
    '''
    Convert a 2D numpy boolean array to a mode '1' (binary) PIL image.

    Input: 
        img_arr : 2D numpy boolean array representing a binary image.
    
    Output:
        img : PIL mode '1' (binary) image.
    '''
    assert img_arr.dtype == np.bool, 'Wrong numpy dtype, expected "np.bool" but got "{}"'.format(img_arr.dtype)
    assert img_arr.ndim == 2, 'Wrong numpy dimension, expected "2" but got "{}" with shape {}'.format(img_arr.ndim, img_arr.shape)

    size = img_arr.shape[::-1]
    databytes = np.packbits(img_arr, axis=1)
    return Image.frombytes(mode='1', size=size, data=databytes)

def numpy_to_pil_grayscale(img_arr: NDArray[np.uint8]):
    '''
    Convert a 2D numpy uint8 array to a mode 'L' (grayscale) PIL image.

    Input: 
        img_arr : 2D numpy uint8 array representing a grayscale image.
    
    Output:
        img : PIL mode 'L' (grayscale) image.
    '''
    assert img_arr.dtype == np.uint8, 'Wrong numpy dtype, expected "np.uint8" but got "{}"'.format(img_arr.dtype)
    assert img_arr.ndim == 2, 'Wrong numpy dimension, expected "2" but got "{}" with shape {}'.format(img_arr.ndim, img_arr.shape)

    # Iteration of creating PIL image tile by tile
    height, width = img_arr.shape
    iters = np.uint8(np.ceil([height / PIL_TILE_SIZE, width / PIL_TILE_SIZE]))

    img = Image.new('L', (width, height))
    for row in range(iters[0]):
        for col in range(iters[1]):
            # Get start and end pixel location
            start_r, start_c = row * PIL_TILE_SIZE, col * PIL_TILE_SIZE
            end_r, end_c = min(height, start_r + PIL_TILE_SIZE), min(width, start_c + PIL_TILE_SIZE)

            # Paste the tile into image
            img_tile = Image.fromarray(img_arr[start_r:end_r, start_c:end_c], 'L')
            img.paste(img_tile, (start_c, start_r))
            del img_tile
    return img

def numpy_to_pil_rgb(img_arr: NDArray[np.uint8]):
    '''
    Convert a 3D numpy uint8 array to a mode 'RGB' (8-bit each channel) PIL image.

    Input: 
        img_arr : 3D numpy uint8 array representing an RGB image, (..., ..., 3).
    
    Output:
        img : PIL mode 'RGB' (8-bit each channel) image.
    '''
    assert img_arr.dtype == np.uint8, 'Wrong numpy dtype, expected "np.uint8" but got "{}"'.format(img_arr.dtype)
    assert img_arr.ndim == 3, 'Wrong numpy dimension, expected "3" but got "{}" with shape {}'.format(img_arr.ndim, img_arr.shape)
    assert img_arr.shape[2] == 3, 'Wrong number of channels, expected "3" but got with shape {}'.format(img_arr.shape)

    # Iteration of creating PIL image tile by tile
    height, width, _ = img_arr.shape
    iters = np.uint8(np.ceil([height / PIL_TILE_SIZE, width / PIL_TILE_SIZE]))

    img = Image.new('RGB', (width, height))
    for row in range(iters[0]):
        for col in range(iters[1]):
            # Get start and end pixel location
            start_r, start_c = row * PIL_TILE_SIZE, col * PIL_TILE_SIZE
            end_r, end_c = min(height, start_r + PIL_TILE_SIZE), min(width, start_c + PIL_TILE_SIZE)

            # Paste the tile into image
            img_tile = Image.fromarray(img_arr[start_r:end_r, start_c:end_c], 'RGB')
            img.paste(img_tile, (start_c, start_r))
            del img_tile
    return img
