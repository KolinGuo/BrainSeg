"""Dataset Related Functions/Classes"""
# pylint: disable=invalid-name

import os
import glob
import sys
import re
from typing import Tuple, List, Dict

from PIL import Image
import numpy as np
from tqdm import tqdm
from tensorflow.keras.utils import Sequence

from utils.svs_to_png import svs_to_numpy
from utils.numpy_pil_helper import numpy_to_pil_binary, numpy_to_pil_palette

Image.MAX_IMAGE_PIXELS = None
VAL_PROP = 1/3

def save_predicted_masks(mask_arr: "NDArray[np.uint8]", save_dir: str,
                         svs_name: str) -> None:
    """
    Save the predicted masks

    Inputs:
        mask_arr : whole image masks, (height, width), [0, 1, 2] = [back, gray, white]
        save_dir : saving directory
        svs_name : name of this WSI
    """
    save_mask_path = os.path.join(save_dir, svs_name+'.png')
    save_back_mask_path = os.path.join(save_dir, svs_name+'-Background.png')
    save_gray_mask_path = os.path.join(save_dir, svs_name+'-Gray.png')
    save_white_mask_path = os.path.join(save_dir, svs_name+'-White.png')

    mask_img = numpy_to_pil_palette(mask_arr)
    # For label 0, leave as black color
    # For label 1, set to cyan color: R0G255B255
    # For label 2, set to yellow color: R255G255B0
    mask_img.putpalette([0, 0, 0, 0, 255, 255, 255, 255, 0])
    mask_img.save(save_mask_path)
    del mask_img

    back_mask_img = numpy_to_pil_binary(mask_arr == 0)
    gray_mask_img = numpy_to_pil_binary(mask_arr == 1)
    white_mask_img = numpy_to_pil_binary(mask_arr == 2)
    back_mask_img.save(save_back_mask_path)
    gray_mask_img.save(save_gray_mask_path)
    white_mask_img.save(save_white_mask_path)
    del back_mask_img, gray_mask_img, white_mask_img

def reconstruct_predicted_masks(patch_masks: "NDArray[np.float32]",
                                patch_coords: "NDArray[int]") \
            -> "NDArray[np.uint8]":
    """
    Reconstruct whole image masks from patch_masks

    Inputs:
        patch_masks  : predicted mask in patches, (num_patches, patch_size, patch_size, 3)
        patch_coords : patch coordinate in original WSI
    Output:
        mask_arr : whole image masks, (height, width), [0, 1, 2] = [back, gray, white]
    """
    patch_masks = np.argmax(patch_masks, axis=-1)
    assert patch_masks.shape[0] == patch_coords.shape[0], "Num_patches mismatch"

    # The end_r and end_c of last patch_coords are height and width
    height, width = patch_coords[-1, -2:]
    num_patches = patch_coords.shape[0]
    mask_arr = np.zeros((height, width), dtype=np.uint8)

    # Reconstruct mask_arr based on patch_coords
    for i in range(num_patches):
        start_r, start_c, end_r, end_c = patch_coords[i, :]
        mask_arr[start_r:end_r, start_c:end_c] = patch_masks[i, ...]

    return mask_arr

def generate_norm_patches(svs_path: str, patch_size: int) \
        -> Tuple["NDArray[np.float32]", "NDArray[int]"]:
    """
    Generate normalized patches of given svs file

    Inputs:
        svs_path   : svs file path
        patch_size : patch size
    Output:
        patches      : normalized patches, (num_patches, patch_size, patch_size, 3)
        patch_coords : patch coordinate in original WSI
    """
    # pylint: disable=too-many-locals
    svs_img_arr = svs_to_numpy(svs_path)
    height, width, _ = svs_img_arr.shape

    iters = np.ceil([height / patch_size, width / patch_size]).astype('int')
    patches = np.zeros((iters[0]*iters[1], patch_size, patch_size, 3),
                       dtype=np.float32)
    # start_r, start_c, end_r, end_c
    patch_coords = np.zeros((iters[0]*iters[1], 4), dtype=int)
    idx = 0
    for row in range(iters[0]):
        for col in range(iters[1]):
            # Get start and end pixel location
            if row != iters[0] - 1:
                start_r = row * patch_size
                end_r = start_r + patch_size
            else:
                start_r = height - patch_size
                end_r = height
            if col != iters[1] - 1:
                start_c = col * patch_size
                end_c = start_c + patch_size
            else:
                start_c = width - patch_size
                end_c = width

            # Cut patches
            patches[idx, ...] = svs_img_arr[start_r:end_r, start_c:end_c, :]
            patch_coords[idx, ...] = [start_r, start_c, end_r, end_c]
            idx += 1
    del svs_img_arr
    # pylint: enable=too-many-locals
    return patches/255.0, patch_coords

def extract_patches(img_arr, patch_size, keep_as_view=True):
    """Extract patches from an svs image array"""
    M, N, D = img_arr.shape
    p0, p1 = patch_size

    if keep_as_view:
        return img_arr.reshape(M//p0, p0, N//p1, p1, D).swapaxes(1, 2)

    return img_arr.reshape(M//p0, p0, N//p1, p1, D).swapaxes(1, 2)\
            .reshape(-1, p0, p1, D)

def generate_predict_dataset(data_dirs: List[str], patch_size: int) \
        -> Tuple[str, str]:
    """
    Generate a prediction dataset

    Inputs:
        data_dirs  : list of .svs directory paths
        patch_size : patch size
    Outputs:
        svs_paths : list of .svs file paths
        save_dir  : patch saving directory path
    """
    # pylint: disable=too-many-locals
    print('Generating predict dataset')

    ##### Get WSI paths #####
    # Convert to abspath
    data_dirs = [os.path.abspath(d) for d in data_dirs]
    # Glob data .svs filepaths
    svs_paths = sorted([p for d in data_dirs
                        for p in glob.glob(os.path.join(d, "*AB*.svs"))])
    print(f'\n\tFound {len(svs_paths)} WSIs in {data_dirs}')

    ##### Generate Patches #####
    save_dir = os.path.join(os.path.dirname(data_dirs[0]), f'patches_{patch_size}')
    print(f'Generating patches of size {patch_size}x{patch_size} \n\t'
          f'for {len(svs_paths)} WSIs \n\t'
          f'saving at "{save_dir}"')

    for svs_path in tqdm(svs_paths):
        svs_name = svs_path.split('/')[-1].replace('.svs', '')

        save_img_dir = os.path.join(save_dir, 'images', svs_name)
        if not os.path.exists(save_img_dir):
            os.makedirs(save_img_dir)
        else:   # Skip if already generated for this WSI
            continue

        # Convert svs to numpy array
        svs_img_arr = svs_to_numpy(svs_path)
        height, width, _ = svs_img_arr.shape

        iters = np.ceil([height / patch_size, width / patch_size]).astype('int')
        for row in range(iters[0]):
            for col in range(iters[1]):
                # Get start and end pixel location
                if row != iters[0] - 1:
                    start_r = row * patch_size
                    end_r = start_r + patch_size
                else:
                    start_r = height - patch_size
                    end_r = height
                if col != iters[1] - 1:
                    start_c = col * patch_size
                    end_c = start_c + patch_size
                else:
                    start_c = width - patch_size
                    end_c = width

                # Cut patches
                svs_patch = Image.fromarray(
                    svs_img_arr[start_r:end_r, start_c:end_c], 'RGB')

                # Save patches
                svs_patch.save(os.path.join(save_img_dir, svs_name \
                        + f'_({start_r},{start_c},{end_r},{end_c}).png'))
        del svs_img_arr
    # pylint: enable=too-many-locals
    return svs_paths, save_dir

def get_patch_paths_and_coords(patch_dir: str, svs_name: str) \
        -> Tuple[List[str], "NDArray[int]"]:
    """
    Returns the patch_paths and patch_coords for this svs

    Inputs:
        patch_dir : patch directory path
        svs_name  : svs_name to look for
    Outputs:
        patch_paths  : list of patch file paths
        patch_coords : ndarray of patch coordinates
    """
    svs_patch_dir = os.path.join(patch_dir, 'images', svs_name)
    patch_paths = sorted(glob.glob(os.path.join(svs_patch_dir, "*.png")),
                         key=lambda p: \
            [int(s) for s in re.split(r'\(|,|\)', p.split('/')[-1])[1:-1]])

    patch_coords = np.array([[int(s)
                              for s in re.split(r'\(|,|\)', p.split('/')[-1])[1:-1]]
                             for p in patch_paths], dtype=int)
    return patch_paths, patch_coords

def compute_class_weight(save_svs_file: str, class_freq_dataset='train') \
        -> "NDArray[np.float]":
    """
    Computing class weight

    Inputs:
        save_svs_file : .txt file path of dataset descriptions and class freq
        class_freq_dataset : the dataset of class frequency to use
    Outputs:
        class_weights : ndarray of class weights at each class index
    """
    search_pattern = ''
    if class_freq_dataset == 'train':
        search_pattern = 'Train Class Frequency'
    elif class_freq_dataset == 'val':
        search_pattern = 'Val Class Frequency'
    elif class_freq_dataset == 'total':
        search_pattern = 'Total Class Frequency'
    else:
        raise ValueError('Unknown class_freq_dataset')

    class_freq = []
    with open(save_svs_file, 'r') as f:
        for line in f:
            if line.startswith(search_pattern):
                class_freq_str = line.strip().split(':')[1].strip(' []')
                class_freq = [int(x) for x in class_freq_str.split()]
                break
    assert class_freq, f'No "Train Class Frequency" found in "{save_svs_file}"'
    class_freq = np.array(class_freq)

    # Inverse class frequency
    class_freq = 1. / class_freq

    class_weights = class_freq / class_freq.sum()
    #class_weights = {k:v for k, v in enumerate(class_weights)}
    return class_weights

def _get_svs_and_truth_paths(data_dir_AD: str, data_dir_control: str) \
        -> Tuple[List[str], List[str], List[str], List[str]]:
    """Get svs and groundtruth paths, remove those svs without groundtruths"""
    ##### Get WSI paths #####
    # Glob data .svs filepaths
    svs_AD_paths = sorted(glob.glob(os.path.join(data_dir_AD, "*AB*.svs")))
    svs_control_paths = sorted(glob.glob(os.path.join(data_dir_control, "*AB*.svs")))

    print(f'Found {len(svs_AD_paths)} AD WSIs '
          f'and {len(svs_control_paths)} control WSIs')

    ##### Get groundtruth paths #####
    # Glob strings for mask .png files
    glob_strs = {
        'gray'   : "*-Gray.png",
        'white'  : "*-White.png",
        'back'   : "*-Background.png"
        }

    # Check if groundtruth is available
    truth_AD_paths, truth_control_paths = [], []
    data_AD_no_truth, data_control_no_truth = [], []
    for i, svs_path in enumerate(svs_AD_paths):
        svs_name = svs_path.split('/')[-1].replace('.svs', '')
        glob_masks = {k: glob.glob(
            os.path.join(data_dir_AD, 'groundtruth', svs_name+v))
                      for k, v in glob_strs.items()}
        if all(glob_masks.values()):
            truth_AD_paths.append(glob_masks)
        else:
            data_AD_no_truth.append(i)

    for i, svs_path in enumerate(svs_control_paths):
        svs_name = svs_path.split('/')[-1].replace('.svs', '')
        glob_masks = {k: glob.glob(
            os.path.join(data_dir_control, 'groundtruth', svs_name+v))
                      for k, v in glob_strs.items()}
        if all(glob_masks.values()):
            truth_control_paths.append(glob_masks)
        else:
            data_control_no_truth.append(i)

    # Remove svs without groundtruths
    svs_AD_removed = [p for i, p in enumerate(svs_AD_paths)
                      if i in data_AD_no_truth]
    svs_control_removed = [p for i, p in enumerate(svs_control_paths)
                           if i in data_control_no_truth]

    svs_AD_paths = [p for i, p in enumerate(svs_AD_paths)
                    if i not in data_AD_no_truth]
    svs_control_paths = [p for i, p in enumerate(svs_control_paths)
                         if i not in data_control_no_truth]

    if svs_AD_removed:
        print(f"\t{len(svs_AD_removed)} AD WSIs don't have groundtruths\n"
              f"\t{[p.split('/')[-1] for p in svs_AD_removed]}\n"
              "\tWon't include them in dataset")
    if svs_control_removed:
        print(f"\t{len(svs_control_removed)} control WSIs don't have groundtruths\n"
              f"\t{[p.split('/')[-1] for p in svs_control_removed]}\n"
              "\tWon't include them in dataset")

    return svs_AD_paths, svs_control_paths, truth_AD_paths, truth_control_paths

def _generate_patches(save_dir: str, patch_size: int,
                      svs_paths: List[str], truth_paths: List[str]) \
        -> Tuple[bool, "NDArray[int]"]:
    """Generate patches for svs and groundtruth"""
    # pylint: disable=too-many-locals
    # pylint: disable=too-many-statements
    found_new_svs = False
    total_class_freq = []
    for i, svs_path in enumerate(tqdm(svs_paths)):
        # Get corresponding groundtruth path
        truth_path = truth_paths[i]
        truth_back_path = truth_path['back'][0]  # Label 0
        truth_gray_path = truth_path['gray'][0]  # Label 1
        truth_white_path = truth_path['white'][0] # Label 2

        svs_name = svs_path.split('/')[-1].replace('.svs', '')

        save_img_dir = os.path.join(save_dir, 'images', svs_name)
        save_mask_dir = os.path.join(save_dir, 'masks', svs_name)
        save_class_freq_file = os.path.join(save_mask_dir, 'class_freq.npy')
        if not os.path.exists(save_img_dir):
            os.makedirs(save_img_dir)
        if not os.path.exists(save_mask_dir):
            os.makedirs(save_mask_dir)
        else:   # Skip if already generated for this WSI
            class_freq = np.load(save_class_freq_file)
            total_class_freq += [list(class_freq)]
            continue

        found_new_svs = True

        # Convert svs to numpy array
        svs_img_arr = svs_to_numpy(svs_path)
        truth_back_arr = np.array(Image.open(truth_back_path))
        truth_gray_arr = np.array(Image.open(truth_gray_path))
        truth_white_arr = np.array(Image.open(truth_white_path))
        height, width = truth_back_arr.shape
        assert svs_img_arr.shape[0:-1] == truth_back_arr.shape
        assert svs_img_arr.shape[0:-1] == truth_gray_arr.shape
        assert svs_img_arr.shape[0:-1] == truth_white_arr.shape

        mask_arr = np.zeros_like(truth_back_arr, dtype='uint8')
        mask_arr[truth_gray_arr] = 1
        mask_arr[truth_white_arr] = 2
        class_freq = \
            [truth_back_arr.sum(), truth_gray_arr.sum(), truth_white_arr.sum()]
        class_freq = np.array(class_freq, dtype='int')
        del truth_back_arr, truth_gray_arr, truth_white_arr

        iters = np.ceil([height / patch_size, width / patch_size]).astype('int')
        for row in range(iters[0]):
            for col in range(iters[1]):
                # Get start and end pixel location
                if row != iters[0] - 1:
                    start_r = row * patch_size
                    end_r = start_r + patch_size
                else:
                    start_r = height - patch_size
                    end_r = height
                if col != iters[1] - 1:
                    start_c = col * patch_size
                    end_c = start_c + patch_size
                else:
                    start_c = width - patch_size
                    end_c = width

                # Cut patches
                svs_patch = Image.fromarray(
                    svs_img_arr[start_r:end_r, start_c:end_c], 'RGB')
                # Save mask_patch using P (palette) mode to save space
                mask_patch = Image.fromarray(
                    mask_arr[start_r:end_r, start_c:end_c], 'P')
                mask_patch.putpalette([0, 0, 0, 135, 98, 122, 106, 99, 251])

                # Save patches
                svs_patch.save(os.path.join(save_img_dir, svs_name \
                        + f'_({start_r},{start_c},{end_r},{end_c}).png'))
                mask_patch.save(os.path.join(save_mask_dir, svs_name \
                        + f'_({start_r},{start_c},{end_r},{end_c}).png'),
                                transparency=0)
        del svs_img_arr, mask_arr
        # Save class_freq
        np.save(save_class_freq_file, class_freq)
        total_class_freq += [list(class_freq)]

    # Convert to numpy array
    total_class_freq = np.array(total_class_freq, dtype='int')

    # pylint: enable=too-many-locals
    # pylint: enable=too-many-statements
    return found_new_svs, total_class_freq

def _split_trainval_dataset(svs_AD_paths: List[str], svs_control_paths: List[str]) \
        -> Dict[int, "NDArray[int]"]:
    """Split and save data into train/val dataset"""
    # Divide into train/val sets
    val_AD_count = int(np.ceil(len(svs_AD_paths) * VAL_PROP))
    train_AD_count = len(svs_AD_paths) - val_AD_count
    val_control_count = int(np.ceil(len(svs_control_paths) * VAL_PROP))
    train_control_count = len(svs_control_paths) - val_control_count

    # Randomly select svs into train/val sets
    AD_idx = np.random.permutation(len(svs_AD_paths))
    control_idx = np.random.permutation(len(svs_control_paths))
    train_AD_idx = AD_idx[0:train_AD_count]
    val_AD_idx = AD_idx[train_AD_count:]
    train_control_idx = control_idx[0:train_control_count]
    val_control_idx = control_idx[train_control_count:]

    print(f'WSI AD: train = {train_AD_count}, val = {val_AD_count}')
    print(f'WSI Control: train = {train_control_count}, val = {val_control_count}')

    return {0: train_AD_idx, 1: val_AD_idx, 2: train_control_idx, 3: val_control_idx}

def _save_trainval_dataset(save_svs_file: str,
                           svs_AD_paths: List[str],
                           svs_control_paths: List[str],
                           train_val_idx: Dict[int, "NDArray[int]"],
                           total_class_freq: "NDArray[int]") \
        -> Tuple[List[str], List[str]]:
    """Save train/val dataset"""
    train_AD_paths = [p for i, p in enumerate(svs_AD_paths)
                      if i in train_val_idx[0]]
    val_AD_paths = [p for i, p in enumerate(svs_AD_paths)
                    if i in train_val_idx[1]]
    train_control_paths = [p for i, p in enumerate(svs_control_paths)
                           if i in train_val_idx[2]]
    val_control_paths = [p for i, p in enumerate(svs_control_paths)
                         if i in train_val_idx[3]]
    # For indexing total_class_freq
    train_idx = np.append(train_val_idx[0], (train_val_idx[2]+ len(svs_AD_paths)))
    val_idx = np.append(train_val_idx[1], (train_val_idx[3]+ len(svs_AD_paths)))
    with open(save_svs_file, 'w') as f:
        # Write dataset split
        f.write(f'AD WSI: Train = {len(train_AD_paths)}, '
                f'Validation = {len(val_AD_paths)}\n')
        f.write('\tTrain: \n\t\t{}\n\tVal: \n\t\t{}\n'\
                .format('\n\t\t'.join(train_AD_paths),
                        '\n\t\t'.join(val_AD_paths)))
        f.write(f'Control WSI: Train = {len(train_control_paths)}, '
                f'Validation = {len(val_control_paths)}\n')
        f.write('\tTrain: \n\t\t{}\n\tVal: \n\t\t{}\n'\
                .format('\n\t\t'.join(train_control_paths),
                        '\n\t\t'.join(val_control_paths)))
        # Write class frequency
        f.write('Class Frequency: [back, gray, white]\n')
        for i, svs_path in enumerate(svs_AD_paths + svs_control_paths):
            svs_name = svs_path.split('/')[-1].replace('.svs', '')
            f.write(f'\t{svs_name}: {total_class_freq[i]}\n')
        f.write('Train Class Frequency: {}\n'\
                .format(np.sum(total_class_freq[train_idx], axis=0)))
        f.write('Val Class Frequency: {}\n'\
                .format(np.sum(total_class_freq[val_idx], axis=0)))
        f.write('Total Class Frequency: {}\n'\
                .format(np.sum(total_class_freq, axis=0)))
    print(f'Train/Val svs files see "{save_svs_file}"\n')

    return train_AD_paths + train_control_paths, val_AD_paths + val_control_paths

def _save_patches_paths(save_dir: str,
                        save_train_file: str,
                        save_val_file: str,
                        train_paths: List[str],
                        val_paths: List[str]) -> None:
    """Save paths to patches as .npy"""
    # Get list of tuple of patches, list(image_path, mask_path)
    train_patches, val_patches = [], []     # type: ignore
    for svs_path in train_paths:
        svs_name = svs_path.split('/')[-1].replace('.svs', '')
        save_img_dir = os.path.join(save_dir, 'images', svs_name, "*.png")
        save_mask_dir = os.path.join(save_dir, 'masks', svs_name, "*.png")

        train_patches += zip(sorted(glob.glob(save_img_dir)),
                             sorted(glob.glob(save_mask_dir)))

    for svs_path in val_paths:
        svs_name = svs_path.split('/')[-1].replace('.svs', '')
        save_img_dir = os.path.join(save_dir, 'images', svs_name, "*.png")
        save_mask_dir = os.path.join(save_dir, 'masks', svs_name, "*.png")

        val_patches += zip(sorted(glob.glob(save_img_dir)),
                           sorted(glob.glob(save_mask_dir)))

    # Shuffle the dataset and save
    train_patch_paths = np.array(train_patches)
    val_patch_paths = np.array(val_patches)
    np.random.shuffle(train_patch_paths)
    np.random.shuffle(val_patch_paths)
    np.save(save_train_file, train_patch_paths)
    np.save(save_val_file, val_patch_paths)

    print(f'Patch Dataset: train = {train_patch_paths.shape}, val = {val_patch_paths.shape}')
    print(f'Dataset saved as "{save_train_file}" and "{save_val_file}"')

def generate_dataset(data_dir_AD: str, data_dir_control: str,
                     patch_size: int, force_regenerate=False) \
        -> Tuple[str, str, str]:
    """
    Generate a dataset

    Inputs:
        data_dir_AD      : AD .svs directory path
        data_dir_control : control .svs directory path
        patch_size       : patch size
        force_regenerate : whether to force regenerate dataset or not
    Outputs:
        save_svs_file   : .txt file path of dataset descriptions
        save_train_file : .npy file path of training data
        save_val_file   : .npy file path of validation data
    """
    # pylint: disable=too-many-locals
    print('Generating train/eval dataset')

    # Convert to abspath
    data_dir_AD = os.path.abspath(data_dir_AD)
    data_dir_control = os.path.abspath(data_dir_control)

    ##### Get svs and groundtruth paths #####
    svs_AD_paths, svs_control_paths, truth_AD_paths, truth_control_paths \
            = _get_svs_and_truth_paths(data_dir_AD, data_dir_control)

    ##### Generate Patches #####
    save_dir = os.path.join(os.path.dirname(data_dir_AD), f'patches_{patch_size}')
    print(f'Generating patches of size {patch_size}x{patch_size} \n\t'
          f'for {len(svs_AD_paths)} AD WSIs '
          f'and {len(svs_control_paths)} control WSIs\n\t'
          f'saving at "{save_dir}"')

    found_new_svs, total_class_freq \
            = _generate_patches(save_dir, patch_size,
                                svs_AD_paths + svs_control_paths,
                                truth_AD_paths + truth_control_paths)

    ##### Split into train and validation dataset #####
    save_svs_file = os.path.join(save_dir, 'dataset.txt')
    save_train_file = os.path.join(save_dir, 'train.npy')
    save_val_file = os.path.join(save_dir, 'val.npy')

    # Return if no need for regenerate dataset
    if not force_regenerate and not found_new_svs \
            and os.path.exists(save_svs_file) \
            and os.path.exists(save_train_file) \
            and os.path.exists(save_val_file):
        print("Found existing dataset, won't regenerate it")
        return save_svs_file, save_train_file, save_val_file

    train_val_idx = _split_trainval_dataset(svs_AD_paths, svs_control_paths)

    ##### Save train and validation dataset #####
    train_paths, val_paths \
            = _save_trainval_dataset(save_svs_file,
                                     svs_AD_paths,
                                     svs_control_paths,
                                     train_val_idx,
                                     total_class_freq)

    _save_patches_paths(save_dir,
                        save_train_file,
                        save_val_file,
                        train_paths,
                        val_paths)

    # pylint: enable=too-many-locals
    return save_svs_file, save_train_file, save_val_file

def _split_trainval_five_fold_dataset(fold_num: int) \
        -> Dict[int, "NDArray[int]"]:
    """Split and save data into train/val five-fold dataset"""
    # Train/Val Dataset Division
    # AD: ['4299', '4092', | '4077', '3777', | '4391', '4195', |
    #      '4471', '4450', '4463', | '4256', '4160', '4107']
    # CONTROL: ['4993', '4972', | '4971', '5010', | '4967', '4992', |
    #           '4944', | '4894']
    TRAIN_WSI = {1: ['4077', '3777', '4971', '5010', '4391', '4195', '4967', '4992',
                     '4471', '4450', '4463', '4944', '4256', '4160', '4107', '4894'],
                 2: ['4299', '4092', '4993', '4972', '4391', '4195', '4967', '4992',
                     '4471', '4450', '4463', '4944', '4256', '4160', '4107', '4894'],
                 3: ['4299', '4092', '4993', '4972', '4077', '3777', '4971', '5010',
                     '4471', '4450', '4463', '4944', '4256', '4160', '4107', '4894'],
                 4: ['4299', '4092', '4993', '4972', '4077', '3777', '4971', '5010',
                     '4391', '4195', '4967', '4992', '4256', '4160', '4107', '4894'],
                 5: ['4299', '4092', '4993', '4972', '4077', '3777', '4971', '5010',
                     '4391', '4195', '4967', '4992', '4471', '4450', '4463', '4944'],
                 6: ['4299', '4092', '4993', '4972',
                     '4077', '3777', '4971', '5010', '4391', '4195', '4967', '4992',
                     '4471', '4450', '4463', '4944', '4256', '4160', '4107', '4894']}
    VAL_WSI = {1: ['4299', '4092', '4993', '4972'],
               2: ['4077', '3777', '4971', '5010'],
               3: ['4391', '4195', '4967', '4992'],
               4: ['4471', '4450', '4463', '4944'],
               5: ['4256', '4160', '4107', '4894'],
               6: []}

    AD_WSI = ['3777', '4077', '4092', '4107', '4160', '4195', '4256', '4299', '4391',
              '4450', '4463', '4471', '4553', '4626', '4672', '4675', '4691', '4695']
    CONTROL_WSI = ['4894', '4907', '4944', '4945', '4967', '4971',
                   '4972', '4992', '4993', '5010', '5015', '5029']

    train_AD_idx = [AD_WSI.index(n) for n in TRAIN_WSI[fold_num] if n in AD_WSI]
    train_control_idx = [CONTROL_WSI.index(n) for n in TRAIN_WSI[fold_num] if n in CONTROL_WSI]
    val_AD_idx = [AD_WSI.index(n) for n in VAL_WSI[fold_num] if n in AD_WSI]
    val_control_idx = [CONTROL_WSI.index(n) for n in VAL_WSI[fold_num] if n in CONTROL_WSI]

    return {0: np.array(train_AD_idx, dtype='int'),
            1: np.array(val_AD_idx, dtype='int'),
            2: np.array(train_control_idx, dtype='int'),
            3: np.array(val_control_idx, dtype='int')}

def generate_five_fold_dataset(data_dir_AD: str, data_dir_control: str,
                               patch_size: int, fold_num: int) \
        -> Tuple[str, str, str]:
    """
    Generate a five-fold dataset

    Inputs:
        data_dir_AD      : AD .svs directory path
        data_dir_control : control .svs directory path
        patch_size       : patch size
        fold_num         : fold number
    Outputs:
        save_svs_file   : .txt file path of dataset descriptions
        save_train_file : .npy file path of training data
        save_val_file   : .npy file path of validation data
    """
    # pylint: disable=too-many-locals
    print(f'Generating train/eval five-fold dataset: fold #{fold_num}')

    # Convert to abspath
    data_dir_AD = os.path.abspath(data_dir_AD)
    data_dir_control = os.path.abspath(data_dir_control)

    ##### Get svs and groundtruth paths #####
    svs_AD_paths, svs_control_paths, truth_AD_paths, truth_control_paths \
            = _get_svs_and_truth_paths(data_dir_AD, data_dir_control)

    ##### Generate Patches #####
    save_dir = os.path.join(os.path.dirname(data_dir_AD), f'patches_{patch_size}')

    _, total_class_freq \
            = _generate_patches(save_dir, patch_size,
                                svs_AD_paths + svs_control_paths,
                                truth_AD_paths + truth_control_paths)

    ##### Split into train and validation dataset #####
    save_svs_file = os.path.join(save_dir, 'dataset.txt')
    save_train_file = os.path.join(save_dir, 'train.npy')
    save_val_file = os.path.join(save_dir, 'val.npy')

    train_val_idx = _split_trainval_five_fold_dataset(fold_num)

    ##### Save train and validation dataset #####
    train_paths, val_paths \
            = _save_trainval_dataset(save_svs_file,
                                     svs_AD_paths,
                                     svs_control_paths,
                                     train_val_idx,
                                     total_class_freq)

    _save_patches_paths(save_dir,
                        save_train_file,
                        save_val_file,
                        train_paths,
                        val_paths)

    # pylint: enable=too-many-locals
    return save_svs_file, save_train_file, save_val_file

class BrainSegPredictSequence(Sequence):
    """BrainSeg Sequence for predict"""
    def __init__(self, image_paths: "NDArray[str]", batch_size: int) -> None:
        self.image_paths = image_paths
        self.batch_size = batch_size

    def __len__(self) -> int:
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def __getitem__(self, idx: int) -> "NDArray[np.float32]":
        batch_x = self.image_paths[idx * self.batch_size :
                                   (idx+1) * self.batch_size]
        return np.array([np.array(Image.open(p)) for p in batch_x],
                        dtype=np.float32) / 255.0

class BrainSegSequence(Sequence):
    """BrainSeg Sequence for train/eval"""
    def __init__(self, image_paths: "NDArray[str]",
                 mask_paths: "NDArray[str]", batch_size: int) -> None:
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.batch_size = batch_size

    def __len__(self) -> int:
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def __getitem__(self, idx: int) -> Tuple["NDArray[np.float32]", "NDArray[np.float32]"]:
        batch_x = self.image_paths[idx * self.batch_size :
                                   (idx+1) * self.batch_size]
        batch_y = self.mask_paths[idx * self.batch_size :
                                  (idx+1) * self.batch_size]
        return np.array([np.array(Image.open(p)) for p in batch_x],
                        dtype=np.float32) / 255.0, \
               np.array([np.array(Image.open(p)) for p in batch_y],
                        dtype=np.int32)

def load_dataset(save_svs_file: str, save_train_file: str, save_val_file: str,
                 batch_size: int) \
        -> Tuple[Sequence, Sequence, "NDArray[np.float]"]:
    """Load the train/eval dataset"""
    # Compute class weights
    class_weight = compute_class_weight(save_svs_file)

    # Load train_paths and val_paths
    train_paths = np.load(save_train_file)
    val_paths = np.load(save_val_file)

    # Create train_dataset and val_dataset
    train_dataset, val_dataset = None, None
    if train_paths.size != 0:
        train_dataset = BrainSegSequence(train_paths[:, 0], train_paths[:, 1],
                                         batch_size)
    if val_paths.size != 0:
        val_dataset = BrainSegSequence(val_paths[:, 0], val_paths[:, 1],
                                       batch_size)

    return train_dataset, val_dataset, class_weight

if __name__ == '__main__':
    ### For Testing ###
    generate_dataset(sys.argv[1], sys.argv[2], int(sys.argv[3]))
# pylint: enable=invalid-name
