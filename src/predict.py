#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
"""Predicting Script"""
import os
import gc
import argparse
from typing import List
import argcomplete

import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.mixed_precision import experimental as mixed_precision

from networks.dataset import generate_predict_dataset, \
        get_patch_paths_and_coords, reconstruct_predicted_masks, \
        save_predicted_masks, BrainSegPredictSequence
from networks.models.models import get_model, load_whole_model
from utils.compute_mask_accuracy import ComputeMaskAccuracy

def get_parser() -> argparse.ArgumentParser:
    """Get the argparse parser for this script"""
    main_parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Predicting\n\t')

    ckpt_parser = main_parser.add_argument_group('Model checkpoint configurations')
    ckpt_parser.add_argument(
        "ckpt_filepath", type=str,
        help="Checkpoint filepath to load and resume predicting from "
             "e.g. ./cp-001-50.51.ckpt.index")
    ckpt_parser.add_argument(
        "--ckpt-weights-only", action='store_true',
        help="Checkpoints will only save the model weights (Default: False)")
    ckpt_parser.add_argument(
        '--model',
        choices=['UNet_No_Pad', 'UNet_No_Pad_3Layer',
                 'UNet_Zero_Pad', 'UNet_Zero_Pad_3Layer',
                 'UNet_Zero_Pad_2019O',
                 'FCN'],
        help="Network model used for predicting")

    dataset_parser = main_parser.add_argument_group('Dataset configurations')
    dataset_parser.add_argument(
        "svs_dirs", type=str, nargs='+',
        help="Directories of svs files (e.g. data/box_Ab data/box_control)")
    dataset_parser.add_argument(
        "--patch-size", type=int, default=1024,
        help="Patch size (Default: 1024)")

    predict_parser = main_parser.add_argument_group('Predicting configurations')
    predict_parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size of patches")
    predict_parser.add_argument(
        "--save-dir", type=str, default="/BrainSeg/data/outputs",
        help="Output directory (Default: /BrainSeg/data/outputs/model_name)")
    predict_parser.add_argument(
        "--compute-accuracy", action='store_true',
        help="Compute accuracy after predicting the dataset (Default: False)")

    testing_parser = main_parser.add_argument_group('Testing configurations')
    testing_parser.add_argument(
        "--test-svs-idx", type=int, default=-1,
        help="Test predicting svs index (Testing only, don't modify)")
    testing_parser.add_argument(
        "--predict-one-round", action='store_true',
        help="Only predict one round (Default: False; Testing only, don't modify)")

    return main_parser

def set_keras_mixed_precision_policy(policy_name: str) -> None:
    """Set tf.keras mixed precision"""
    policy = mixed_precision.Policy(policy_name)
    mixed_precision.set_policy(policy)

def predict_svs(model: keras.Model, args,
                svs_name: str, svs_patch_paths: List[str],
                patch_coords: "NDArray[int]") \
        -> "NDArray[np.uint8]":
    """Use model to predict for this svs"""
    # Get number of patches
    num_patches = patch_coords.shape[0]
    print(f'\n\t{svs_name}: generated {num_patches} patches')

    # Limit one-time predict to 5 GB data
    patch_masks = np.zeros(
        (num_patches, args.patch_size, args.patch_size, 3), dtype=np.float32)
    num_patch_per_round = 5*1024**3 // \
            (np.prod(patch_masks.shape[1:]) * patch_masks.itemsize)
    num_rounds = int(np.ceil(num_patches / num_patch_per_round)) \
            if not args.predict_one_round else 1
    print(f'\tBeginning {num_rounds} rounds of prediction')
    for i in range(num_rounds):
        start_p = i * num_patch_per_round
        end_p = min(num_patches, start_p + num_patch_per_round)

        # Create a data Sequence
        patches_dataset = BrainSegPredictSequence(
            svs_patch_paths[start_p:end_p], args.batch_size)

        # Pass to model for prediction
        patch_masks[start_p:end_p, ...] \
                = model.predict(patches_dataset,
                                verbose=1)
        # TODO: Switch to tf.data

    # Reconstruct whole image from patch_masks
    mask_arr = reconstruct_predicted_masks(patch_masks, patch_coords)
    del patch_masks, patch_coords

    return mask_arr

def predict(args):
    """Start predicting based on args input"""
    # Check if GPU is available
    print("\nNum GPUs Available: %d\n"\
          % (len(tf.config.list_physical_devices('GPU'))))

    # Set tf.keras mixed precision to float16
    set_keras_mixed_precision_policy('mixed_float16')

    # Create dataset
    svs_paths, patch_dir \
            = generate_predict_dataset(args.svs_dirs, args.patch_size)

    # Create network model
    if args.ckpt_weights_only:
        if args.ckpt_filepath.endswith('.index'):   # Get rid of the suffix
            args.ckpt_filepath = args.ckpt_filepath.replace('.index', '')
        model = get_model(args.model)
        model.load_weights(args.ckpt_filepath).assert_existing_objects_matched()
        print('Model weights loaded')
    else:
        model = load_whole_model(args.ckpt_filepath)
        print('Whole model (weights + optimizer state) loaded')

    # Create output directory
    args.save_dir = os.path.join(os.path.abspath(args.save_dir), model.name)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    print(f'\tSaving predicted mask to "{args.save_dir}"')

    # If in testing mode, test only the svs of that index
    if args.test_svs_idx != -1:
        svs_paths = [svs_paths[args.test_svs_idx]]

    success_count = 0
    for svs_path in tqdm(svs_paths):
        svs_name = svs_path.split('/')[-1].replace('.svs', '')

        # Load corresponding patches
        svs_patch_paths, patch_coords \
                = get_patch_paths_and_coords(patch_dir, svs_name)

        # Use model to predict this svs
        mask_arr = predict_svs(model, args,
                               svs_name, svs_patch_paths, patch_coords)

        # Save predicted masks
        save_predicted_masks(mask_arr, args.save_dir, svs_name)
        del mask_arr

        success_count += 1
        gc.collect()

    # Prompt for compute_mask_accuracy
    print('\nOut of %d WSIs, \n\t%d were successfully processed'
          % (len(svs_paths), success_count))

    # If run compute_mask_accuracy.py at the end
    if args.compute_accuracy:
        compute_acc_parser = ComputeMaskAccuracy.get_parser()
        args = compute_acc_parser.parse_args([args.save_dir] \
            + [os.path.join(os.path.abspath(d), 'groundtruth') for d in args.svs_dirs])
        ComputeMaskAccuracy(args)
    else:
        print(f'To compute mask accuracy, please run compute_mask_accuracy.py {args.save_dir}')


if __name__ == '__main__':
    parser = get_parser()
    argcomplete.autocomplete(parser)
    predict_args = parser.parse_args()

    predict(predict_args)
