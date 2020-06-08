#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
"""Training Script"""
from datetime import datetime
import os
import io
import argparse
from typing import List
import argcomplete

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import optimizers, metrics, callbacks
from tensorflow.keras.mixed_precision import experimental as mixed_precision

from networks.dataset import generate_dataset, load_dataset
from networks.models.models import get_model, load_whole_model
from networks.metrics import SparseIoU, SparseMeanIoU, SparseConfusionMatrix
from networks.losses import get_loss_func

def get_parser() -> argparse.ArgumentParser:
    """Get the argparse parser for this script"""
    main_parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Training\n\t')

    model_parser = main_parser.add_argument_group('Model configurations')
    model_parser.add_argument(
        'model',
        choices=['UNet_No_Pad', 'UNet_No_Pad_3Layer',
                 'UNet_Zero_Pad', 'UNet_Zero_Pad_3Layer',
                 'FCN'],
        help="Network model used for training")

    dataset_parser = main_parser.add_argument_group('Dataset configurations')
    dataset_parser.add_argument(
        "--data-dir-AD", type=str, default='/BrainSeg/data/box_Ab',
        help="Directory of AD svs files (Default: data/box_Ab)")
    dataset_parser.add_argument(
        "--data-dir-control", type=str, default='/BrainSeg/data/box_control',
        help="Directory of control svs files (Default: data/box_control)")
    dataset_parser.add_argument(
        "--patch-size", type=int, default=1024,
        help="Patch size (Default: 1024)")
    dataset_parser.add_argument(
        "--val-subsplits", type=int, default=1,
        help="Number to divide total number of val data for each epoch")

    train_parser = main_parser.add_argument_group('Training configurations')
    train_parser.add_argument(
        '--loss-func',
        choices=['SCCE', 'BSCCE', 'Sparse_Focal', 'Balanced_Sparse_Focal'],
        default='Sparse_Focal',
        help="Loss functions for training")
    train_parser.add_argument(
        "--focal-loss-gamma", type=float, default=2.0,
        help="Gamma parameter for focal loss (Default: 2.0)")
    train_parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size of patches")
    train_parser.add_argument(
        "--num-epochs", type=int, default=20,
        help="Number of training epochs")
    train_parser.add_argument(
        "--ckpt-weights-only", action='store_true',
        help="Checkpoints will only save the model weights (Default: False)")
    train_parser.add_argument(
        "--ckpt-dir", type=str, default='/BrainSeg/checkpoints',
        help="Directory for saving/loading checkpoints")
    train_parser.add_argument(
        "--ckpt-filepath", type=str, default=None,
        help="Checkpoint filepath to load and resume training from "
        "e.g. ./cp-001-50.51.ckpt.index")
    train_parser.add_argument(
        "--log-dir", type=str, default='/BrainSeg/tf_logs',
        help="Directory for saving tensorboard logs")
    train_parser.add_argument(
        "--file-suffix", type=str,
        default=datetime.now().strftime("%Y%m%d_%H%M%S"),
        help="Suffix for ckpt file and log file (Default: current timestamp)")

    testing_parser = main_parser.add_argument_group('Testing configurations')
    testing_parser.add_argument(
        "--steps-per-epoch", type=int, default=-1,
        help="Training steps per epoch (Testing only, don't modify)")
    testing_parser.add_argument(
        "--val-steps", type=int, default=-1,
        help="Validation steps (Testing only, don't modify)")

    return main_parser

def set_keras_mixed_precision_policy(policy_name: str) -> None:
    """Set tf.keras mixed precision"""
    policy = mixed_precision.Policy(policy_name)
    mixed_precision.set_policy(policy)

def log_configs(log_dir: str, dataset_filepath: str,
                train_dataset, val_dataset, args) -> None:
    """Log configuration of this training session"""
    writer = tf.summary.create_file_writer(log_dir + "/config")
    with writer.as_default():
        tf.summary.text("Dataset", open(dataset_filepath).read(), step=0)
        for key, value in vars(args).items():
            tf.summary.text(str(key), str(value), step=0)
        tf.summary.text(
            "Train_Batches_per_epoch",
            str(len(train_dataset)), step=0)
        tf.summary.text(
            "Val_Batches_per_epoch",
            str(len(val_dataset) // args.val_subsplits), step=0)
        writer.flush()

def plot_confusion_matrix(cmat, class_names):
    """Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
        cmat (array, shape = [n, n]): a confusion matrix of integer classes
        class_names (array, shape = [n]): String names of the integer classes
    """
    figure = plt.figure(figsize=(5, 5), dpi=100)
    plt.imshow(cmat, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    prop = np.around(cmat.astype('float') / cmat.sum(axis=1)[:, np.newaxis],
                     decimals=4) * 100

    # Use white text if squares are dark; otherwise black.
    threshold = cmat.max() / 2.
    for i in range(cmat.shape[0]):
        for j in range(cmat.shape[1]):
            color = "white" if cmat[i, j] > threshold else "black"
            plt.text(j, i, '%.2f%%\n%.2e' % (prop[i, j], cmat[i, j]),
                     horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure

def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.io.decode_png(buf.getvalue())
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image

def get_cm_callback(log_dir: str, class_names: List[str]) -> callbacks.Callback:
    """Get the confusion matrix callback for plotting"""
    def log_confusion_matrix(epoch, logs):
        """Use tf.summary.image to plot confusion matrix"""
        figure = plot_confusion_matrix(logs['cm'], class_names=class_names)
        cm_image = plot_to_image(figure)
        with cm_image_writer.as_default():
            tf.summary.image("Train Confusion Matrix", cm_image, step=epoch)

        figure = plot_confusion_matrix(logs['val_cm'], class_names=class_names)
        cm_image = plot_to_image(figure)
        with cm_image_writer.as_default():
            tf.summary.image("Val Confusion Matrix", cm_image, step=epoch)

    cm_image_writer = tf.summary.create_file_writer(log_dir + "/cm")
    return callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)

def train(args) -> None:
    """Start training based on args input"""
    # Check if GPU is available
    print("\nNum GPUs Available: %d\n"\
          % (len(tf.config.list_physical_devices('GPU'))))

    # Set tf.keras mixed precision to float16
    set_keras_mixed_precision_policy('mixed_float16')

    # Create dataset
    save_svs_file, save_train_file, save_val_file \
            = generate_dataset(args.data_dir_AD, args.data_dir_control,
                               args.patch_size, force_regenerate=False)

    # Load dataset
    train_dataset, val_dataset, class_weight \
            = load_dataset(save_svs_file, save_train_file, save_val_file,
                           args.batch_size)

    # Create network model
    model = get_model(args.model)
    #model.summary(120)
    #print(keras.backend.floatx())

    class_names = ['Background', 'Gray Matter', 'White Matter']
    model.compile(optimizer=optimizers.Adam(),
                  loss=get_loss_func(args.loss_func, class_weight,
                                     gamma=args.focal_loss_gamma),
                  metrics=[metrics.SparseCategoricalAccuracy(),
                           SparseMeanIoU(num_classes=3, name='IoU/Mean'),
                           SparseConfusionMatrix(num_classes=3, name='cm')] \
            + SparseIoU.get_iou_metrics(num_classes=3, class_names=class_names))

    # Create another checkpoint/log folder for model.name and timestamp
    args.ckpt_dir = os.path.join(args.ckpt_dir,
                                 model.name+'-'+args.file_suffix)
    args.log_dir = os.path.join(args.log_dir, 'fit',
                                model.name+'-'+args.file_suffix)

    # Check if resume from training
    initial_epoch = 0
    if args.ckpt_filepath is not None:
        if args.ckpt_filepath.endswith('.index'):
            args.ckpt_filepath \
                    = args.ckpt_filepath.replace('.index', '')

        if args.ckpt_weights_only:
            model.load_weights(args.ckpt_filepath)\
                    .assert_existing_objects_matched()
            print('Model weights loaded')
        else:
            model = load_whole_model(args.ckpt_filepath)
            print('Whole model (weights + optimizer state) loaded')

        initial_epoch = int(args.ckpt_filepath.split('/')[-1]\
                .split('-')[1])
        # Save in same checkpoint_dir but different log_dir (add current time)
        args.ckpt_dir = os.path.abspath(
            os.path.dirname(args.ckpt_filepath))
        args.log_dir = args.ckpt_dir.replace(
            'checkpoints', 'tf_logs/fit') + f'-retrain_{args.file_suffix}'

    # Write configurations to log_dir
    log_configs(args.log_dir, save_svs_file, train_dataset, val_dataset, args)

    # Create checkpoint directory
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    # Create log directory
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    # Create a callback that saves the model's weights every 1 epoch
    ckpt_path = os.path.join(
        args.ckpt_dir, 'cp-{epoch:03d}-{val_IoU/Mean:.4f}.ckpt')
    cp_callback = callbacks.ModelCheckpoint(
        filepath=ckpt_path,
        verbose=1,
        save_weights_only=args.ckpt_weights_only,
        save_freq='epoch')

    # Create a TensorBoard callback
    tb_callback = callbacks.TensorBoard(
        log_dir=args.log_dir,
        histogram_freq=1,
        write_graph=True,
        write_images=False,
        update_freq='batch',
        profile_batch='100, 120')

    # Create a Lambda callback for plotting confusion matrix
    cm_callback = get_cm_callback(args.log_dir, class_names)

    # Create a TerminateOnNaN callback
    nan_callback = callbacks.TerminateOnNaN()

    model.fit(
        train_dataset,
        epochs=args.num_epochs,
        steps_per_epoch=len(train_dataset) \
                if args.steps_per_epoch == -1 else args.steps_per_epoch,
        initial_epoch=initial_epoch,
        validation_data=val_dataset,
        validation_steps=len(val_dataset) // args.val_subsplits \
                if args.val_steps == -1 else args.val_steps,
        callbacks=[cp_callback, tb_callback, nan_callback, cm_callback],
        workers=os.cpu_count(),
        use_multiprocessing=True)

    print('Training finished!')


if __name__ == '__main__':
    parser = get_parser()
    argcomplete.autocomplete(parser)
    train_args = parser.parse_args()

    train(train_args)
