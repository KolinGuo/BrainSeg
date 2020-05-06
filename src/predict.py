#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
from datetime import datetime
import os, glob, gc
import argparse, argcomplete
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers, losses, metrics, callbacks
from tensorflow.keras.mixed_precision import experimental as mixed_precision

from train import SparseMeanIoU
from models.FCN import fcn_model
from utils.dataset import generate_norm_patches, reconstruct_predicted_masks, \
        save_predicted_masks

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description='Predicting\n\t')

    ckpt_parser = parser.add_argument_group(
            'Model checkpoint configurations')
    ckpt_parser.add_argument("ckpt_filepath", type=str, 
            help="Checkpoint filepath to load and resume predicting from "
            "e.g. ./cp-001-50.51.ckpt.index")
    ckpt_parser.add_argument("--ckpt-weights-only", action='store_true',
            help="Checkpoints will only save the model weights (Default: False)")
    ckpt_parser.add_argument('--model', choices=['UNet', 'FCN'],
            help="Network model used for predicting")

    dataset_parser = parser.add_argument_group(
            'Dataset configurations')
    dataset_parser.add_argument("svs_dirs", type=str, nargs='+',
            help="Directories of svs files (e.g. data/box_Ab data/box_control)")
    dataset_parser.add_argument("--patch-size", type=int, default=1024,
            help="Patch size (Default: 1024)")

    predict_parser = parser.add_argument_group(
            'Predicting configurations')
    predict_parser.add_argument("--batch-size", type=int, default=32,
            help="Batch size of patches")
    predict_parser.add_argument("--save-dir", type=str, 
            default="/BrainSeg/data/outputs",
            help="Output directory (Default: /BrainSeg/data/outputs/model_name)")

    return parser

def predict(args):
    # Check if GPU is available
    print("Num GPUs Available: %d", len(tf.config.list_physical_devices('GPU')))

    # Set tf.keras mixed precision to float16
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_policy(policy)

    # Create dataset
    svs_paths = sorted([p for d in args.svs_dirs 
        for p in glob.glob(os.path.join(os.path.abspath(d), "*AB*.svs"))])
    print(f'\tFound {len(svs_paths)} WSIs in {args.svs_dirs}')

    # Create network model
    if args.ckpt_weights_only:
        if args.model == 'UNet':
            model = unet_model_zero_pad(output_channels=3)
        elif args.model == 'FCN':
            model = fcn_model(classes=3, bn=True)
        latest = tf.train.latest_checkpoint(args.ckpt_filepath)
        model.load_weights(latest)
        print('Model weights loaded')
    else:
        model = keras.models.load_model(args.ckpt_filepath, 
                custom_objects={'SparseMeanIoU': SparseMeanIoU})
        print('Full model (weights + optimizer state) loaded')

    # Create output directory
    args.save_dir = os.path.join(os.path.abspath(args.save_dir), model.name)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    print(f'\tSaving predicted mask to "{args.save_dir}"')

    success_count = 0
    for i, svs_path in enumerate(tqdm(svs_paths)):
        svs_name = svs_path.split('/')[-1].replace('.svs', '')
        # Generate patches from svs
        patches, patch_coords = generate_norm_patches(svs_path, args.patch_size)

        print(f'\t{svs_name}: generated {patches.shape[0]} patches')

        # Pass to model for prediction
        patch_masks = model.predict(patches, 
                batch_size=args.batch_size,
                verbose=1)
        del patches

        # Reconstruct whole image from patch_masks
        mask_arr = reconstruct_predicted_masks(patch_masks, patch_coords)
        del patch_masks, patch_coords

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
