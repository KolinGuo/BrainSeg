import numpy as np

def makeODLUT(max_val: int, n_entries: int=256) -> "NDArray[float]":
    """
    Create an optical density lookup table, normalizing to the specified background value.

    Inputs:
        max_val   : background (white value).
        n_entries : number of values to include in the lookup table.
    Output:
        out : numpy array, optical density lookup table.
    """
    vals = np.arange(n_entries)     # pixel value 0-255

    OD_LUT = np.maximum(0, -np.log10(np.maximum(vals, 1)/max_val))

    return OD_LUT

def color_deconv(rgb_img: "NDArray[np.uint8]", stain_mat: "NDArray[int]",
        stain_bk_rgb: "NDArray[np.uint8]", stain_idx: int=None) -> "NDArray[float]":
    """
    RGB to stain color space conversion using color deconvolution.
    Modified based on QuPath's implementation
    https://github.com/qupath/qupath/blob/d893d081524542ed7a27922949bd38ba6d795d30/qupath-core/src/main/java/qupath/lib/color/ColorTransformer.java#L398

    Inputs:
        rgb_img      : 3-D numpy array of shape (..., ..., 3), image in RGB format.
        stain_mat    : numpy array of shape (3, 3), stain matrix.
        stain_bk_rgb : numpy array of shape (3, ), background RGB values, 0-255.
        stain_idx    : integer, the selected stain index, default is None (first two stains, H&E).
    Output:
        out : numpy array, the image in stain color channel.
    """

    # Make OD lookup table for faster computation
    OD_LUT_red   = makeODLUT(stain_bk_rgb[0])
    OD_LUT_green = makeODLUT(stain_bk_rgb[1])
    OD_LUT_blue  = makeODLUT(stain_bk_rgb[2])

    # Inverse the stain matrix
    stain_mat = np.linalg.inv(stain_mat)

    # Compute OD values using OD_LUTs
    od_rgb_img = np.zeros_like(rgb_img, dtype='float64')
    od_rgb_img[:, :, 0] = OD_LUT_red[rgb_img[:, :, 0]]
    od_rgb_img[:, :, 1] = OD_LUT_green[rgb_img[:, :, 1]]
    od_rgb_img[:, :, 2] = OD_LUT_blue[rgb_img[:, :, 2]]

    # Convolute with stain_mat
    if stain_idx is None:   # get first two stains (H&E)
        out_shape = (rgb_img.shape[0], rgb_img.shape[1], 2)
        od_rgb_img = np.matmul(np.reshape(od_rgb_img, (-1, 3)), (stain_mat[:,0:2]), dtype='float64')
    else:   # get the selected stain
        out_shape = (rgb_img.shape[0], rgb_img.shape[1])    # type: ignore
        od_rgb_img = np.matmul(np.reshape(od_rgb_img, (-1, 3)), (stain_mat[:,stain_idx]), dtype='float64')

    return np.reshape(od_rgb_img, out_shape)

if __name__ == '__main__':
    ### For testing ###
    rgb = np.array([[[201, 211, 230]]], dtype='uint8')
    stain_mat = np.array([[0.45353514056487343, 0.5886228507331238, 0.6692002808334822], [0.2628197298091816, 0.5014792096359046, 0.8242841694015344], [0.5785660804362569, -0.7655927158056977, 0.28129892298742126]])
    stain_bk_rgb = np.array([242, 242, 241])
    stain_idx = 1

    print('Stain_mat: ')
    print(stain_mat)
    print('Stain_bk_rgb: ')
    print(stain_bk_rgb)
    print('Stain_idx: ')
    print(stain_idx)
    
    print(color_deconv(rgb, stain_mat, stain_bk_rgb, stain_idx))
