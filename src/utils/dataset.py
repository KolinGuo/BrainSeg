import os, glob, sys
import numpy as np
from tqdm import tqdm
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from svs_to_png import svs_to_numpy

def generate_dataset(data_dir_AD, data_dir_control, patch_size):
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

    ##### Save paths to patches as .npy #####

    ##### Split into train and validation dataset #####


if __name__ == '__main__':
    ### For Testing ###
    generate_dataset(sys.argv[1], sys.argv[2], int(sys.argv[3]))
