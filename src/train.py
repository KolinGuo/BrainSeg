#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
from datetime import datetime
import os
import argparse, argcomplete
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers, losses, metrics, callbacks
from tensorflow.keras.mixed_precision import experimental as mixed_precision

from models.UNet import unet_model_zero_pad
from utils.dataset import generate_dataset, BrainSegSequence

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description='Training\n\t')

    parser.add_argument('model', choices=['UNet'],
            help="Network model used for training")

    dataset_parser = parser.add_argument_group(
            'Dataset configurations')
    dataset_parser.add_argument("--data-dir-AD", type=str, 
            default='/BrainSeg/data/box_Ab', 
            help="Directory of AD svs files (Default: data/box_Ab)")
    dataset_parser.add_argument("--data-dir-control", type=str, 
            default='/BrainSeg/data/box_control', 
            help="Directory of control svs files (Default: data/box_control)")
    dataset_parser.add_argument("--patch-size", type=int, default=1024,
            help="Patch size (Default: 1024)")

    train_parser = parser.add_argument_group(
            'Training configurations')
    train_parser.add_argument("--batch-size", type=int, default=32,
            help="Batch size of patches")
    train_parser.add_argument("--num-epochs", type=int, default=20,
            help="Number of training epochs")
    train_parser.add_argument("--checkpoint-weights-only", action='store_true',
            help="Save the model weights only as checkpoints (Default: False)")
    train_parser.add_argument("--checkpoint-dir", type=str, 
            default='/BrainSeg/checkpoints', 
            help="Directory for saving/loading checkpoints")
    train_parser.add_argument("--checkpoint-filepath", type=str, 
            default=None, 
            help="Checkpoint filepath to load and resume training from "
            "e.g. ./cp-001-50.51.ckpt.index")
    train_parser.add_argument("--log-dir", type=str, 
            default='/BrainSeg/tf_logs', 
            help="Directory for saving tensorboard logs")

    #mask_parser = parser.add_argument_group(
    #        'select the kinds of masks to compute accuracy')
    #mask_parser.add_argument("-b", "--background", action='store_true',
    #        help="Compute accuracy for background masks only")
    #mask_parser.add_argument("-g", "--gray", action='store_true',
    #        help="Compute accuracy for gray matter masks only")
    #mask_parser.add_argument("-w", "--white", action='store_true',
    #        help="Compute accuracy for white matter masks only")
    #mask_parser.add_argument("-t", "--tissue", action='store_true',
    #        help="Compute accuracy for tissue masks only")

    #metric_parser = parser.add_argument_group(
    #        'select the kinds of metrics to compute accuracy')
    #metric_parser.add_argument("-p", "--pixel-accuracy", action='store_true',
    #        help="Compute accuracy using pixel accuracy only")
    #metric_parser.add_argument("-ma", "--mean-accuracy", action='store_true',
    #        help="Compute accuracy using mean accuracy only")
    #metric_parser.add_argument("-i", "--iou", action='store_true',
    #        help="Compute accuracy using IoU only")
    #metric_parser.add_argument("-mi", "--mean-iou", action='store_true',
    #        help="Compute accuracy using mean IoU only")
    #metric_parser.add_argument("-fi", "--frequency-iou", action='store_true',
    #        help="Compute accuracy using frequency weighted IoU only")
    #metric_parser.add_argument("-f", "--f1-score", action='store_true',
    #        help="Compute accuracy using F1 score only")

    return parser

class SparseMeanIoU(metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        return super().update_state(y_true, y_pred, sample_weight)

def train(args):
    # Check if GPU is available
    print("Num GPUs Available: %d", len(tf.config.list_physical_devices('GPU')))

    # Set tf.keras mixed precision to float16
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_policy(policy)

    # Create dataset
    save_train_file, save_val_file \
            = generate_dataset(args.data_dir_AD, args.data_dir_control, 
                    args.patch_size)

    train_paths = np.load(save_train_file)
    val_paths = np.load(save_val_file)

    train_dataset = BrainSegSequence(train_paths[:, 0], train_paths[:, 1], 
            args.batch_size)
    val_dataset = BrainSegSequence(val_paths[:, 0], val_paths[:, 1], 
            args.batch_size)

    # Create network model
    if args.model == 'UNet':
        model = unet_model_zero_pad(output_channels=3)
    #model.summary(120)
    #print(keras.backend.floatx())

    model.compile(optimizer=optimizers.Adam(),
            loss=losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[metrics.SparseCategoricalAccuracy(),
                SparseMeanIoU(num_classes=3, name='meaniou')])

    # Create another checkpoint/log folder for model.name and timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, 
            model.name+'-'+timestamp)
    args.log_dir = os.path.join(args.log_dir, 'fit', 
            model.name+'-'+timestamp)

    # Check if resume from training
    initial_epoch = 0
    if args.checkpoint_filepath is not None:
        if args.checkpoint_filepath.endswith('.index'):
            args.checkpoint_filepath \
                    = args.checkpoint_filepath.replace('.index', '')

        if args.checkpoint_weights_only:
            model.load_weights(args.checkpoint_filepath)\
                    .assert_existing_objects_matched()
            print('Model weights loaded')
        else:
            model = keras.models.load_model(args.checkpoint_filepath)
            print('Full model (weights + optimizer state) loaded')

        initial_epoch = int(args.checkpoint_filepath.split('/')[-1]\
                .split('-')[1])
        # Save in same checkpoint_dir but different log_dir (add current time)
        args.checkpoint_dir = os.path.abspath(
                os.path.dirname(args.checkpoint_filepath))
        args.log_dir = args.checkpoint_dir.replace(
                'checkpoints', 'tf_logs/fit') + f'-retrain_{timestamp}'

    # Write configurations to log_dir
    writer = tf.summary.create_file_writer(args.log_dir + "/config")
    with writer.as_default():
        for key, value in vars(args).items():
            tf.summary.text(str(key), str(value), step=0)
        tf.summary.text("Train_Batches_per_epoch", str(len(train_dataset)), step=0)
        tf.summary.text("Val_Batches_per_epoch", str(len(val_dataset)), step=0)
        writer.flush()

    # Create checkpoint directory
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    # Create log directory
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    # Create a callback that saves the model's weights every 1 epoch
    checkpoint_path = os.path.join(args.checkpoint_dir, 
            'cp-{epoch:03d}-{val_acc:.2f}.ckpt')
    cp_callback = callbacks.ModelCheckpoint(filepath=checkpoint_path,
            verbose=1, 
            save_weights_only=args.checkpoint_weights_only,
            save_freq='epoch')

    # Create a TensorBoard callback
    tb_callback = callbacks.TensorBoard(log_dir=args.log_dir, 
            histogram_freq=1,
            write_graph=True,
            write_images=True,
            update_freq='batch',
            profile_batch=2)

    model.fit(train_dataset, 
            epochs=args.num_epochs,
            steps_per_epoch=len(train_dataset),
            initial_epoch=initial_epoch,
            validation_data=val_dataset,
            validation_steps=len(val_dataset),
            callbacks=[cp_callback, tb_callback])

    print('Training finished!')


if __name__ == '__main__':
    parser = get_parser()
    argcomplete.autocomplete(parser)
    train_args = parser.parse_args()

    train(train_args)
