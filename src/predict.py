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

from models.FCN import fcn_model
from models.UNet import unet_model_zero_pad
from models.metrics import SparseIoU, SparseMeanIoU, SparseConfusionMatrix
from utils.dataset import generate_predict_dataset, get_patch_paths_and_coords, \
        reconstruct_predicted_masks, save_predicted_masks, \
        BrainSegPredictSequence

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
        '--model', choices=['UNet', 'FCN'],
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
    num_rounds = int(np.ceil(num_patches / num_patch_per_round))
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
                                verbose=1,
                                workers=os.cpu_count(),
                                use_multiprocessing=True)

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
        if args.model == 'UNet':
            model = unet_model_zero_pad(output_channels=3)
        elif args.model == 'FCN':
            model = fcn_model(classes=3, bn=True)
        model.load_weights(args.ckpt_filepath).assert_existing_objects_matched()
        print('Model weights loaded')
    else:
        model = keras.models.load_model(
            args.ckpt_filepath,
            custom_objects={
                'SparseMeanIoU': SparseMeanIoU,
                'SparseConfusionMatrix': SparseConfusionMatrix,
                'SparseIoU': SparseIoU})
        print('Full model (weights + optimizer state) loaded')

    # Create output directory
    args.save_dir = os.path.join(os.path.abspath(args.save_dir), model.name)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    print(f'\tSaving predicted mask to "{args.save_dir}"')

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
    print(f'To compute mask accuracy, please run compute_mask_accuracy.py {args.save_dir}')


if __name__ == '__main__':
    parser = get_parser()
    argcomplete.autocomplete(parser)
    predict_args = parser.parse_args()

    predict(predict_args)
