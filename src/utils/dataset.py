import os, glob, sys
import numpy as np
from tqdm import tqdm
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from typing import Tuple, List

from .svs_to_png import svs_to_numpy
from .numpy_pil_helper import numpy_to_pil_binary, numpy_to_pil_palette

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
    save_mask_path       = os.path.join(save_dir, svs_name+'.png')
    save_back_mask_path  = os.path.join(save_dir, svs_name+'-Background.png')
    save_gray_mask_path  = os.path.join(save_dir, svs_name+'-Gray.png')
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
        patch_coords: "NDArray[int]") -> "NDArray[np.uint8]":
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
                end_r   = start_r + patch_size
            else:
                start_r = height - patch_size
                end_r   = height
            if col != iters[1] - 1:
                start_c = col * patch_size
                end_c   = start_c + patch_size
            else:
                start_c = width - patch_size
                end_c   = width

            # Cut patches
            patches[idx, ...] = svs_img_arr[start_r:end_r, start_c:end_c, :]
            patch_coords[idx, ...] = [start_r, start_c, end_r, end_c]
            idx += 1

    del svs_img_arr
    return patches/255.0, patch_coords

def extract_patches(img_arr, patch_size, keep_as_view=True):
    M, N, D = img_arr.shape
    p0, p1  = patch_size

    if keep_as_view:
        return img_arr.reshape(M//p0, p0, N//p1, p1, D).swapaxes(1,2)
    else:
        return img_arr.reshape(M//p0, p0, N//p1, p1, D).swapaxes(1,2)\
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
    print('Generating dataset')

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

    for i, svs_path in enumerate(tqdm(svs_paths)):
        svs_name = svs_path.split('/')[-1].replace('.svs', '')

        save_img_dir = os.path.join(save_dir, 'images', svs_name)
        if not os.path.exists(save_img_dir):
            os.makedirs(save_img_dir)
        else:   # Skip if already generated for this WSI
            continue

        # Convert svs to numpy array
        svs_img_arr      = svs_to_numpy(svs_path)
        height, width, _ = svs_img_arr.shape

        iters = np.ceil([height / patch_size, width / patch_size]).astype('int')
        for row in range(iters[0]):
            for col in range(iters[1]):
                # Get start and end pixel location
                if row != iters[0] - 1:
                    start_r = row * patch_size
                    end_r   = start_r + patch_size
                else:
                    start_r = height - patch_size
                    end_r   = height
                if col != iters[1] - 1:
                    start_c = col * patch_size
                    end_c   = start_c + patch_size
                else:
                    start_c = width - patch_size
                    end_c   = width

                # Cut patches
                svs_patch = Image.fromarray(
                        svs_img_arr[start_r:end_r, start_c:end_c], 'RGB')

                # Save patches
                svs_patch.save(os.path.join(save_img_dir, 
                    svs_name+f'_({start_r},{start_c},{end_r},{end_c}).png'))
        del svs_img_arr
    return svs_paths, save_dir

def generate_dataset(data_dir_AD: str, data_dir_control: str, 
        patch_size: int, force_regenerate: bool=False) -> Tuple[str, str, str]:
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
    print('Generating dataset')

    # Convert to abspath
    data_dir_AD = os.path.abspath(data_dir_AD)
    data_dir_control = os.path.abspath(data_dir_control)

    ##### Get WSI paths #####
    # Glob data .svs filepaths
    svs_AD_paths = sorted([p for p in 
        glob.glob(os.path.join(data_dir_AD, "*AB*.svs"))])
    svs_control_paths = sorted([p for p in 
        glob.glob(os.path.join(data_dir_control, "*AB*.svs"))])

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

    ##### Generate Patches #####
    save_dir = os.path.join(os.path.dirname(data_dir_AD), f'patches_{patch_size}')
    print(f'Generating patches of size {patch_size}x{patch_size} \n\t'
            f'for {len(svs_AD_paths)} AD WSIs '
            f'and {len(svs_control_paths)} control WSIs\n\t'
            f'saving at "{save_dir}"')

    found_new_svs = False
    for i, svs_path in enumerate(tqdm(svs_AD_paths + svs_control_paths)):
        # Get corresponding groundtruth path
        truth_paths      = (truth_AD_paths + truth_control_paths)[i]
        truth_back_path  = truth_paths['back'][0]  # Label 0
        truth_gray_path  = truth_paths['gray'][0]  # Label 1
        truth_white_path = truth_paths['white'][0] # Label 2

        svs_name = svs_path.split('/')[-1].replace('.svs', '')

        save_img_dir = os.path.join(save_dir, 'images', svs_name)
        save_mask_dir = os.path.join(save_dir, 'masks', svs_name)
        if not os.path.exists(save_img_dir):
            os.makedirs(save_img_dir)
        if not os.path.exists(save_mask_dir):
            os.makedirs(save_mask_dir)
        else:   # Skip if already generated for this WSI
            continue

        found_new_svs = True

        # Convert svs to numpy array
        svs_img_arr     = svs_to_numpy(svs_path)
        truth_back_arr  = np.array(Image.open(truth_back_path))
        truth_gray_arr  = np.array(Image.open(truth_gray_path))
        truth_white_arr = np.array(Image.open(truth_white_path))
        height, width   = truth_back_arr.shape
        assert svs_img_arr.shape[0:-1] == truth_back_arr.shape
        assert svs_img_arr.shape[0:-1] == truth_gray_arr.shape
        assert svs_img_arr.shape[0:-1] == truth_white_arr.shape

        mask_arr = np.zeros_like(truth_back_arr, dtype='uint8')
        mask_arr[truth_gray_arr] = 1
        mask_arr[truth_white_arr] = 2
        del truth_back_arr, truth_gray_arr, truth_white_arr

        iters = np.ceil([height / patch_size, width / patch_size]).astype('int')
        for row in range(iters[0]):
            for col in range(iters[1]):
                # Get start and end pixel location
                if row != iters[0] - 1:
                    start_r = row * patch_size
                    end_r   = start_r + patch_size
                else:
                    start_r = height - patch_size
                    end_r   = height
                if col != iters[1] - 1:
                    start_c = col * patch_size
                    end_c   = start_c + patch_size
                else:
                    start_c = width - patch_size
                    end_c   = width

                # Cut patches
                svs_patch  = Image.fromarray(
                        svs_img_arr[start_r:end_r, start_c:end_c], 'RGB')
                # Save mask_patch using P (palette) mode to save space
                mask_patch = Image.fromarray(
                        mask_arr[start_r:end_r, start_c:end_c], 'P')
                mask_patch.putpalette([0, 0, 0, 135, 98, 122, 106, 99, 251])

                # Save patches
                svs_patch.save(os.path.join(save_img_dir, 
                    svs_name+f'_({start_r},{start_c},{end_r},{end_c}).png'))
                mask_patch.save(os.path.join(save_mask_dir,
                    svs_name+f'_({start_r},{start_c},{end_r},{end_c}).png'),
                    transparency=0)
        del svs_img_arr, mask_arr

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

    # Divide into train/val sets
    val_AD_count        = int(np.ceil(len(svs_AD_paths) * VAL_PROP))
    train_AD_count      = len(svs_AD_paths) - val_AD_count
    val_control_count   = int(np.ceil(len(svs_control_paths) * VAL_PROP))
    train_control_count = len(svs_control_paths) - val_control_count

    # Randomly select svs into train/val sets
    AD_idx            = np.random.permutation(len(svs_AD_paths))
    control_idx       = np.random.permutation(len(svs_control_paths))
    train_AD_idx      = AD_idx[0:train_AD_count]
    val_AD_idx        = AD_idx[train_AD_count:]
    train_control_idx = control_idx[0:train_control_count]
    val_control_idx   = control_idx[train_control_count:]

    train_AD_paths      = [p for i, p in enumerate(svs_AD_paths) 
            if i in train_AD_idx]
    val_AD_paths        = [p for i, p in enumerate(svs_AD_paths) 
            if i in val_AD_idx]
    train_control_paths = [p for i, p in enumerate(svs_control_paths) 
            if i in train_control_idx]
    val_control_paths   = [p for i, p in enumerate(svs_control_paths) 
            if i in val_control_idx]

    with open(save_svs_file, 'w') as f:
        f.write(f'AD WSI: Train = {train_AD_count}, '
                f'Validation = {val_AD_count}\n')
        f.write('\tTrain: \n\t\t{}\n\tVal: \n\t\t{}\n'\
                .format('\n\t\t'.join(train_AD_paths), 
                    '\n\t\t'.join(val_AD_paths)))
        f.write(f'Control WSI: Train = {train_control_count}, '
                f'Validation = {val_control_count}\n')
        f.write('\tTrain: \n\t\t{}\n\tVal: \n\t\t{}\n'\
                .format('\n\t\t'.join(train_control_paths), 
                    '\n\t\t'.join(val_control_paths)))

    print(f'WSI AD: train = {train_AD_count}, val = {val_AD_count}')
    print(f'WSI Control: train = {train_control_count}, val = {val_control_count}')
    print(f'Train/Val svs files see "{save_svs_file}"\n')

    ##### Save paths to patches as .npy #####
    # Get list of tuple of patches, list(image_path, mask_path)
    train_patches, val_patches = [], []     # type: ignore
    for svs_path in train_AD_paths + train_control_paths:
        svs_name = svs_path.split('/')[-1].replace('.svs', '')
        save_img_dir = os.path.join(save_dir, 'images', svs_name, "*.png")
        save_mask_dir = os.path.join(save_dir, 'masks', svs_name, "*.png")

        train_patches += zip(sorted(glob.glob(save_img_dir)),
                sorted(glob.glob(save_mask_dir)))

    for svs_path in val_AD_paths + val_control_paths:
        svs_name = svs_path.split('/')[-1].replace('.svs', '')
        save_img_dir = os.path.join(save_dir, 'images', svs_name, "*.png")
        save_mask_dir = os.path.join(save_dir, 'masks', svs_name, "*.png")

        val_patches += zip(sorted(glob.glob(save_img_dir)),
                sorted(glob.glob(save_mask_dir)))

    # Shuffle the dataset and save
    train_patch_paths = np.array(train_patches)
    val_patch_paths   = np.array(val_patches)
    np.random.shuffle(train_patch_paths)
    np.random.shuffle(val_patch_paths)
    np.save(save_train_file, train_patch_paths)
    np.save(save_val_file, val_patch_paths)

    print(f'Patch Dataset: train = {train_patch_paths.shape}, val = {val_patch_paths.shape}')
    print(f'Dataset saved as "{save_train_file}" and "{save_val_file}"')

    return save_svs_file, save_train_file, save_val_file

class BrainSegPredictSequence(Sequence):
    def __init__(self, image_paths: "NDArray[str]", batch_size: int) -> None:
        self.image_paths = image_paths
        self.batch_size  = batch_size

    def __len__(self) -> int:
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def __getitem__(self, idx: int) -> "NDArray[np.float32]":
        batch_x = self.image_paths[idx * self.batch_size : 
                (idx+1) * self.batch_size]
        return np.array([np.array(Image.open(p)) for p in batch_x], 
                    dtype=np.float32) / 255.0

class BrainSegSequence(Sequence):
    def __init__(self, image_paths: "NDArray[str]",
            mask_paths: "NDArray[str]", batch_size: int) -> None:
        self.image_paths = image_paths
        self.mask_paths  = mask_paths
        self.batch_size  = batch_size

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
                    dtype=np.float32)

if __name__ == '__main__':
    ### For Testing ###
    generate_dataset(sys.argv[1], sys.argv[2], int(sys.argv[3]))
