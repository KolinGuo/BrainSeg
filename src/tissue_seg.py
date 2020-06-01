"""Tissue Separation Script (Traditional Image Processing Technique)"""

import os
import sys
import glob
import logging
import gc   # Garbage Collector interface
import argparse
from time import time
from datetime import datetime

import numpy as np
from tqdm import tqdm

from utils.svs_to_png import svs_to_numpy
from utils.separate_tissue import separate_tissue

def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    Output:
        out : command line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "input_dir", type=str,
        help="Input Directory of .png files (original WSI image)")
    parser.add_argument(
        "output_dir", type=str,
        help="Output Directory of .png files")

    parser.add_argument(
        "--log_dir", type=str, default='/BrainSeg/data/outputs/logs',
        help="Directory for saving logs")
    log_filename = datetime.now().strftime("%Y%m%d_%H%M%S.log")
    parser.add_argument(
        "--log_filename", type=str, default=log_filename,
        help="Log file name (current timestamp) DON'T MODIFY")

    return parser.parse_args()

def log_args(args: argparse.Namespace) -> None:
    """
    Setup logging and log the input arguments.

    Input:
        args : command line arguments
    """
    # Create summary writer to write summaries to disk
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    # Set up logging to file
    log_filepath = os.path.join(args.log_dir, args.log_filename)
    logging.basicConfig(level=logging.DEBUG,
                        format='[%(asctime)s] [%(name)-30s] [%(levelname)-8s] %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename=log_filepath,
                        filemode='w')

    # define a Handler which writes INFO messages or higher to the sys.stdout
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter('[%(name)-30s] [%(levelname)-8s] %(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)

    # Get logger for training args
    logger = logging.getLogger('args')

    # Write the training arguments
    for key, value in vars(args).items():
        logger.info('{%s: %s}', key, value)

def main(args: argparse.Namespace) -> None:
    """
    Main tissue_sep function

    Input:
        args : command line arguments
    """
    main_logger = logging.getLogger('main')
    time_logger = logging.getLogger('main_timer')

    assert os.path.isdir(args.input_dir), \
        'Input directory "{}" does not exist!'.format(args.input_dir)

    # Make output directory if not exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    img_names = [pathname.split('/')[-1] for pathname in
                 glob.glob(os.path.join(args.input_dir, "*.svs"))]
    img_names = sorted(img_names)

    stain_file_names = [pathname.split('/')[-1] for pathname in
                        glob.glob(os.path.join(args.input_dir, 'stain_contrast', "*.txt"))]
    stain_file_names = sorted(stain_file_names)
    main_logger.info('Inputs: %d .svs WSIs with %d .txt stain_contrast files',
                     len(img_names), len(stain_file_names))

    timer_results = []
    tissue_timer_results = []
    # pylint: disable=invalid-name
    t = tqdm(total=len(img_names), postfix='\n', leave=False)
    # pylint: enable=invalid-name
    for i, img_name in enumerate(img_names):
        # tdqm progress bar
        t.set_description_str("Image " + img_name, refresh=False)

        # Create svs image path and corresponding stain_file path
        img_path = os.path.join(args.input_dir, img_name)
        stain_file_idxs = [i for i, s in enumerate(stain_file_names)
                           if s.startswith(img_name.split('.')[0])]
        if len(stain_file_idxs) != 1:
            print('Found {} matched stain file whose name starts with "{}". Skipping this image...'\
                .format(len(stain_file_idxs), img_name.split('.')[0]))
            t.update()
            continue
        stain_file_idx = stain_file_idxs[0]
        stain_file_path = os.path.join(
            args.input_dir, 'stain_contrast', stain_file_names[stain_file_idx])

        ##### Start the pipeline #####
        timer_result = [time()]
        time_logger.info('Processing image %s ...', img_name)

        # Convert svs to numpy array
        img_arr = svs_to_numpy(img_path)
        timer_result.append(time())
        time_logger.info('svs_to_numpy: %.4f second', timer_result[-1] - timer_result[-2])

        # Get tissue mask
        tissue_mask_arr, tissue_timer_result \
                = separate_tissue(img_arr, stain_file_path, save_tissue_mask=True,
                                  save_tissue_mask_dir=args.output_dir)
        timer_result.append(time())
        time_logger.info('separate_tissue: %.4f second', timer_result[-1] - timer_result[-2])

        # Free up memory
        del img_arr, tissue_mask_arr
        # Manual garbage collect
        main_logger.debug('Collected %s garbages', gc.collect())

        timer_result = np.array(timer_result)
        timer_result = timer_result[1:] - timer_result[:-1]     # type: ignore

        if i == 0:
            timer_results, tissue_timer_results = timer_result, tissue_timer_result
        else:
            timer_results = np.vstack((timer_results, timer_result))
            tissue_timer_results = np.vstack((tissue_timer_results, tissue_timer_result))

        t.update()
    t.close()

    time_logger.info('per_step_avg: %s seconds', np.mean(timer_results, axis=0))
    time_logger.info('per_tissue_avg: %s seconds', np.mean(timer_results, axis=1))
    tissue_time_logger = logging.getLogger('tissue_timer')
    tissue_time_logger.info('per_step_avg: %s seconds', np.mean(tissue_timer_results, axis=0))
    tissue_time_logger.info('per_tissue_avg: %s seconds', np.mean(tissue_timer_results, axis=1))

    print('Done!')

if __name__ == '__main__':
    tissue_sep_args = parse_arguments()
    log_args(tissue_sep_args)
    main(tissue_sep_args)
