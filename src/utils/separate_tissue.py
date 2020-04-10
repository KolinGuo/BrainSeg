import os, glob
import re   # regex split
import numpy as np
from tqdm import tqdm

import cppyy
cppyy.load_library('/BrainSeg/src/gSLICr/build/libgSLICr_lib.so')
cppyy.load_library('/usr/local/cuda/targets/x86_64-linux/lib/libcudart.so')
cppyy.add_include_path('/usr/local/cuda/targets/x86_64-linux/include')   # Add cuda library
cppyy.include('/BrainSeg/src/gSLICr/gSLICr_Lib/gSLICr.h')
from cppyy.gbl import gSLICr
import cppyy.ll

from skimage.segmentation import find_boundaries, mark_boundaries
from skimage.measure import regionprops, label
from skimage.morphology import binary_dilation, convex_hull_image

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from color_deconv import color_deconv
from numpy_pil_helper import *

def parse_stain_mat(file_path):
    '''
    Parse stain matrix given a file path.
    
    Input:
        file_path : A String to a file containing stain matrix and contrast.
    Outputs:
        stain_mat    : 3-by-3 double np matrix, [H; E; Residual].
        stain_bk_rgb : 1-by-3 int np matrix, stain background RGB value.
        stain_idx    : integer, index of selected stain (row), 0/1/2.
        stain_thres  : 1-by-2 double np array, [min, max].
    '''
    assert os.path.isfile(file_path), 'Input file "{}" does not exist!'.format(file_path)

    stain_mat = []
    stain_names = []
    stain_bk_rgb = []
    stain_idx = -1
    stain_thres = []

    with open(file_path, 'r') as f:
        line = f.readline()
        line_count = 1
        while line:
            line = line.strip()
            # Save stain vectors
            # i.e. "Hematoxylin: [0.45353514056487343, -0.5886228507331238, 0.6692002808334822]"
            if line_count >= 3 and line_count <= 5:
                stain_names.append(line.split(':')[0])
                #arr = [float(x) for x in re.findall(r'[-][0-9].[0-9]+', line)]
                arr = [float(s) for s in re.split(' |\[|,|\]', line)[2:7:2]]
                stain_mat.append(arr)
                assert arr != [], 'No stain vectors for "{}"'.format(stain_names[-1])

            # Save background RGB values
            # i.e. "Background RGB: [242, 242, 241]"
            if line_count == 6:
                #arr = [int(x) for x in re.findall(r'[0-9]+', line)]
                arr = [int(s) for s in re.split(' |\[|,|\]', line)[3:8:2]]
                stain_bk_rgb = arr
                assert stain_bk_rgb != [], 'No stain background RGB values'

            # Save selected channel (stain)
            # i.e. "Selected channel: Eosin"
            if line_count == 7:
                selected_stain_name = line.split(': ')[-1]
                for i, s in enumerate(stain_names):
                    if s == selected_stain_name:
                        stain_idx = i
                        break
                assert stain_idx != -1, 'No such stain named "{}"'.format(selected_stain_name)

            # Save min and max threshold
            # i.e. "[Min, Max]: [-0.4276258945465088, 0.5525617003440857]"
            if line_count == 8:
                #arr = [float(x) for x in re.findall(r'[-][0-9].[0-9]*', line)]
                arr = [float(s) for s in re.split(' |\[|,|\]', line)[6:9:2]]
                stain_thres = arr
                assert stain_thres != [], 'No min max threshold for stain'
                break   # finish parsing

            line = f.readline()
            line_count += 1

    return np.asarray(stain_mat), np.asarray(stain_bk_rgb), stain_idx, np.asarray(stain_thres)

def scale_range(arr, new_min, new_max, old_min=None, old_max=None):
    '''
    Scale array to fit new range.
    
    Inputs:
        arr     : numpy array.
        new_min : new minimum.
        new_max : new maximum.
        old_min : old minimum, default is None. Skip arr.max() if known.
        old_max : old maximum, default is None. Skip arr.max() if known.
    Output:
        out : numpy array after scaling, range [new_min, new_max].
    '''
    if old_min is None:
        old_min = arr.min()
    if old_max is None:
        old_max = arr.max()
    return (arr - old_min) * (new_max - new_min) / (old_max - old_min) + new_min

def apply_stain_thres_grayscale(arr, thres_min, thres_max):
    '''
    Apply contrast threshold on stain channel.
    Pixels value < thres_min are set to black, pixels value > thres_max are set to white.
    Output is linearly mapped from [thres_min, thres_max] to [0, 255].
    
    Inputs:
        arr       : numpy float64 array, stain channel image.
        thres_min : double, minimum stain threshold.
        thres_max : double, maximum stain threshold.
    Output:
        out : numpy uint8 array, thresholded stain channel image, range [0, 255].
    '''
    # Old method
    arr[arr < thres_min] = thres_min
    arr[arr > thres_max] = thres_max

    return scale_range(arr, 0, 255, thres_min, thres_max).astype('uint8')

def apply_stain_thres_binary(arr, thres_min, thres_max):
    '''
    Apply contrast threshold on stain channel.
    Pixels thres_min <= value <= thres_max are set to white, others are black.
    Output is binary: [0, 255].
    
    Inputs:
        arr       : numpy float64 array, stain channel image.
        thres_min : double, minimum stain threshold.
        thres_max : double, maximum stain threshold.
    Output:
        out : numpy uint8 array, thresholded stain channel image, range [0, 255].
    '''
    arr[(arr < thres_min) | (arr > thres_max)] = 0
    arr[(arr >= thres_min) & (arr <= thres_max)] = 255

    return arr.astype('uint8')

def run_slic_seg(img, n_segments=80000, n_iter=5, compactness=0.01, enforce_connectivity=True, slic_zero=True):
    """Run gpu SLICO in C++.

    Parameters
    -----------
    img : 2D ndarray
        Input image, must be grayscale. 
    n_segments : int, optional
        The (approximate) number of labels in the segmented output image.
    n_iter : int, optional
        The number of iterations of k-means.
    compactness : float, optional
        Balances color proximity and space proximity. Higher values give
        more weight to space proximity, making superpixel shapes more
        square/cubic. In SLICO mode, this is the initial compactness.
        This parameter depends strongly on image contrast and on the
        shapes of objects in the image. We recommend exploring possible
        values on a log scale, e.g., 0.01, 0.1, 1, 10, 100, before
        refining around a chosen value.
    enforce_connectivity : bool, optional
        Whether the generated segments are connected or not.
    slic_zero : bool, optional
        Run SLIC-zero, the zero-parameter mode of SLIC. [2]_
        
    Returns
    -------
    segments_slic : 2D ndarray
        Integer mask indicating SLIC segment labels.
        
    References
    ----------
    .. [1] Radhakrishna Achanta, Appu Shaji, Kevin Smith, Aurelien Lucchi,
        Pascal Fua, and Sabine Süsstrunk, SLIC Superpixels Compared to
        State-of-the-art Superpixel Methods, TPAMI, May 2012.
    .. [2] http://ivrg.epfl.ch/research/superpixels#SLICO
    """
    height, width = img.shape
    # Create a Run_SLIC C++ class
    run_slic = gSLICr.Run_SLIC()

    # Specify settings
    gslic_settings = run_slic.get_settings()
    gslic_settings = cppyy.bind_object(gslic_settings, gSLICr.objects.settings)
    gslic_settings.img_size.x = width
    gslic_settings.img_size.y = height
    gslic_settings.no_segs = n_segments
    gslic_settings.no_iters = n_iter
    gslic_settings.coh_weight = compactness
    gslic_settings.do_enforce_connectivity = enforce_connectivity
    gslic_settings.slic_zero = slic_zero
    gslic_settings.color_space = gSLICr.GRAY
    gslic_settings.seg_method = gSLICr.GIVEN_NUM
    run_slic.print_settings()

    # Start running gSLIC
    run_slic.run(img)

    # Get the segmentation mask
    segments_slic = run_slic.get_mask()
    segments_slic.reshape((height*width,))
    segments_slic = np.frombuffer(segments_slic, dtype=np.intc, count=height*width).reshape((height, width)).copy()

    # Destruct the C++ class
    run_slic.__destruct__()
    return segments_slic

def slic_mean_intensity_thres(img, segments_slic, thres=60):
    """Thresholding based on SLIC segments' mean intensity.

    Parameters
    -----------
    img : 2D ndarray
        Input image, must be grayscale.
    segments_slic : 2D ndarray
        Integer mask indicating SLIC segment labels.
    thres : int, optional
        Threshold value of mean intensity.

    Returns
    -------
    mask_img : 2D ndarray
        Binary integer mask indicating segment labels.
    """
    mask_img = np.zeros_like(img, dtype='uint8')
    for region in regionprops(segments_slic, img):
        if region.mean_intensity >= thres:
            mask_img.flat[np.ravel_multi_index(region.coords.transpose(), mask_img.shape)] = 255
    return mask_img

def get_connected_comp(mask_img, structure_size=9):
    """Get connected components inside an image.

    Parameters
    -----------
    mask_img : 2D ndarray
        Input image, must be binary.
    structure_size : int, optional
        Structure size for binary dilation before connected component labeling.

    Returns
    -------
    output : 2D ndarray
        Integer mask indicating connected component labels.
    """
    return label(binary_dilation(mask_img, selem=np.ones((structure_size, structure_size), dtype='int')), connectivity=2)

def find_tissue(img, connected_comp, prop=0.2):
    """Find the tissue component within a connected component map.

    Parameters
    -----------
    img : 2D ndarray
        Input image, must be grayscale.
    connected_comp : 2D ndarray
        Integer mask indicating connected component labels.
    prop : float, optional
        Proportional occupied area of the entire image to be considered as tissue.

    Returns
    -------
    tissue_img : 2D ndarray
        Binary integer mask indicating the tissue.
    """
    height, width = img.shape
    area_thres = prop * height * width
    tissue_img = np.zeros_like(img, dtype='uint8')
    for region in regionprops(connected_comp, img):
        if region.area >= area_thres:
            tissue_img.flat[np.ravel_multi_index(region.coords.transpose(), tissue_img.shape)] = 255
    return tissue_img

def find_refine_boundaries(img, segments_slic, tissue_img):
    """Find and refine boundaries of the tissue image using convex hull.

    Parameters
    -----------
    img : 2D ndarray
        Input image, must be grayscale.
    segments_slic : 2D ndarray
        Integer mask indicating SLIC segment labels.
    tissue_img : 2D ndarray
        Binary integer mask indicating the tissue.
        
    Returns
    -------
    tissue_img : 2D ndarray
        Binary integer refined mask indicating the tissue.
    """
    # Find boundaries
    boundaries = find_boundaries(tissue_img, connectivity=2, mode='thick')
    labels = np.unique(segments_slic[boundaries])
    # Convex hull on boundaries
    for region in regionprops(segments_slic, img):
        if region.label in labels:
            min_r, min_c, max_r, max_c = region.bbox
            img_tile = tissue_img[min_r:max_r, min_c:max_c]
            outlier_hull_img = convex_hull_image(img_tile)
            #tissue_img[min_r:max_r, min_c:max_c] = (outlier_hull_img * 255).astype('uint8')
            tissue_img[min_r:max_r, min_c:max_c] = (outlier_hull_img * 1)
    return tissue_img.astype('bool')

def separate_tissue(input_dir, stain_mat_dir, output_dir):
    assert os.path.isdir(input_dir), 'Input directory "{}" does not exist!'.format(input_dir)
    assert os.path.isdir(stain_mat_dir), 'Stain matrix directory "{}" does not exist!'.format(stain_mat_dir)

    # Make output directory if not exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    img_names = [pathname.split('/')[-1] for pathname in glob.glob(os.path.join(input_dir, "*.png"))]
    img_names = sorted(img_names)

    stain_file_names = [pathname.split('/')[-1] for pathname in glob.glob(os.path.join(stain_mat_dir, "*.txt"))]
    stain_file_names = sorted(stain_file_names)
    
    t = tqdm(total=len(img_names), postfix='\n', leave=False)
    for i, img_name in enumerate(img_names):

        # Read in png image
        img = np.array(Image.open(os.path.join(input_dir, img_name)))
        height, width, _ = img.shape

        # tdqm progress bar
        t.set_description_str("Image " + img_name + ', ' + '({}, {})'.format(width, height), refresh=False)

        # Read in stain matrix
        stain_file_idx = [ i for i, s in enumerate(stain_file_names) if s.startswith(img_name.split('.')[0]) ]
        if len(stain_file_idx) != 1:
            print('Found {} matched stain file whose name starts with "{}". Skipping this image...'.format(len(stain_file_idx), img_name.split('.')[0]))
            continue
        stain_file_idx = stain_file_idx[0]
        stain_mat, stain_bk_rgb, stain_idx, stain_thres = parse_stain_mat(os.path.join(stain_mat_dir, stain_file_names[stain_file_idx]))

        # Separate stains
        img = color_deconv(img, stain_mat, stain_bk_rgb, stain_idx)

        # Apply stain threshold
        img = apply_stain_thres_binary(img, stain_thres[0], stain_thres[1])

        # Run gSLICOr
        segments_slic = run_slic_seg(img)

        # Mean Intensity Thresholding
        mask_img = slic_mean_intensity_thres(img, segments_slic)

        # Connected components
        connected_comp = get_connected_comp(mask_img)
        del mask_img

        # Find tissue component
        tissue_img = find_tissue(img, connected_comp)
        del connected_comp

        # Find and refine boundaries
        tissue_img = find_refine_boundaries(img, segments_slic, tissue_img)
        del img, segments_slic

        # Save img
        tissue_img = numpy_to_pil_binary(tissue_img)
        tissue_img.save(os.path.join(output_dir, img_name))

        t.update()
    t.close()

if __name__ == '__main__':
    ### For testing ###
    import argparse
    args = argparse.ArgumentParser()

    args.add_argument("input_dir", type=str, help="Input Directory of .png files (original WSI image)")
    args.add_argument("stain_mat_dir", type=str, help="Directory of .txt files (stain matrices)")
    args.add_argument("output_dir", type=str, help="Output Directory of .png files")
    args = args.parse_args()

    print("Input directory: " + args.input_dir)
    print("Stain matrix directory: " + args.stain_mat_dir)
    print("Output directory: " + args.output_dir)

    separate_tissue(args.input_dir, args.stain_mat_dir, args.output_dir)

    print("Done!")
