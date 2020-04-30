import os, glob, sys
import numpy as np
from tqdm import tqdm
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from tensorflow.keras.utils import Sequence
from typing import Tuple
from nptyping import NDArray

from svs_to_png import svs_to_numpy

VAL_PROP = 1/3

def generate_dataset(data_dir_AD: str, data_dir_control: str, patch_size: int) -> None:
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
    save_train_file = os.path.join(save_dir, 'train.npy')
    save_val_file = os.path.join(save_dir, 'val.npy')

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

class BrainSegSequence(Sequence):
    def __init__(self, image_paths: NDArray[str],
            mask_paths: NDArray[str], batch_size: int):
        self.image_paths = image_paths
        self.mask_paths  = mask_paths
        self.batch_size  = batch_size

    def __len__(self) -> int:
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def __getitem__(self, idx: int) -> Tuple[NDArray[np.uint8], NDArray[np.uint8]]:
        batch_x = self.image_paths[idx * self.batch_size : 
                (idx+1) * self.batch_size]
        batch_y = self.mask_paths[idx * self.batch_size : 
                (idx+1) * self.batch_size]
        return np.array([np.array(Image.open(p)) for p in batch_x]), \
                np.array([np.array(Image.open(p)) for p in batch_y])

if __name__ == '__main__':
    ### For Testing ###
    generate_dataset(sys.argv[1], sys.argv[2], int(sys.argv[3]))
