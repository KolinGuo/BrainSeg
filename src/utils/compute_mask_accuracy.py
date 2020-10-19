#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
from datetime import datetime
from collections import OrderedDict     # for easy saving
from textwrap import dedent             # for string formatting
import os
import sys
import glob
import argparse
import argcomplete
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from typing import List, Dict, TypeVar, Any
from numbers import Real
Number = TypeVar('Number', bound=Real)

class ComputeMaskAccuracy:
    def __init__(self, args: argparse.Namespace):
        # Convert to abspath
        args.truth_dir = [os.path.abspath(p) for p in args.truth_dir]
        args.test_dir = os.path.abspath(args.test_dir)
        # Check if isdir
        assert all([os.path.isdir(p) for p in args.truth_dir]), \
                f'Groundtruth directory {args.truth_dir} does not exist!'
        assert os.path.isdir(args.test_dir), \
                f'Testing directory {args.test_dir} does not exist!'
        self.compute_mask_accuracy(args)

    @staticmethod
    def get_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
                formatter_class=argparse.RawDescriptionHelpFormatter,
                description='Compute Mask Accuracy\n\t'
                'Specify optional arguments to only calculate accuracies\n\t\t'
                'for those masks using those metrics.\n\t'
                'By default the program will automatically determine\n\t\t'
                'based on directory content and use IoU.')

        parser.add_argument("test_dir", type=str, 
                help="Directory of testing .png files")
        parser.add_argument("truth_dir", type=str, nargs='+',
                help="Directories of groundtruth .png files")

        mask_parser = parser.add_argument_group(
                'select the kinds of masks to compute accuracy')
        mask_parser.add_argument("-b", "--background", action='store_true',
                help="Compute accuracy for background masks only")
        mask_parser.add_argument("-g", "--gray", action='store_true',
                help="Compute accuracy for gray matter masks only")
        mask_parser.add_argument("-w", "--white", action='store_true',
                help="Compute accuracy for white matter masks only")
        mask_parser.add_argument("-t", "--tissue", action='store_true',
                help="Compute accuracy for tissue masks only")

        metric_parser = parser.add_argument_group(
                'select the kinds of metrics to compute accuracy')
        metric_parser.add_argument("-p", "--pixel-accuracy", action='store_true',
                help="Compute accuracy using pixel accuracy only")
        metric_parser.add_argument("-ma", "--mean-accuracy", action='store_true',
                help="Compute accuracy using mean accuracy only")
        metric_parser.add_argument("-i", "--iou", action='store_true',
                help="Compute accuracy using IoU only")
        metric_parser.add_argument("-mi", "--mean-iou", action='store_true',
                help="Compute accuracy using mean IoU only")
        metric_parser.add_argument("-fi", "--frequency-iou", action='store_true',
                help="Compute accuracy using frequency weighted IoU only")
        metric_parser.add_argument("-f", "--f1-score", action='store_true',
                help="Compute accuracy using F1 score only")

        return parser

    def compute_mask_accuracy(self, args: argparse.Namespace):
        def _glob_dir(dir_path: str, glob_str: str) -> List[str]:
            return sorted([p for p in glob.glob(os.path.join(dir_path, glob_str))])

        def _glob_dir_exists(dir_path: str, glob_str: str) -> List[str]:
            paths = _glob_dir(dir_path, glob_str)
            assert paths, f'No {glob_str} found in {dir_path}'
            # Convert *-Grey.png to *-Gray.png
            # for f in *-Grey.png; do mv "$f" "${f%%-Grey.png}"-Gray.png; done
            return paths

        def _assert_same_len_values_in_dict(in_dict: Dict):
            first_val = next(iter(in_dict.items()))[1]
            assert all(len(v) == len(first_val) for v in in_dict.values()), \
                    'Not the same number of masks: {}'.format(
                            {k: len(v) for k, v in in_dict.items()} )

        def _get_test_image_paths() -> "OrderedDict[str, List[str]]":
            if any([args.background, args.gray, args.white, args.tissue]):
                test_paths = OrderedDict()
                if args.gray:
                    test_paths['gray'] = _glob_dir_exists(args.test_dir, glob_strs['gray'])
                if args.white:
                    test_paths['white'] = _glob_dir_exists(args.test_dir, glob_strs['white'])
                if args.background:
                    test_paths['back'] = _glob_dir_exists(args.test_dir, glob_strs['back'])
                if args.tissue:
                    test_paths['tissue'] = _glob_dir_exists(args.test_dir, glob_strs['tissue'])
            else:   # automatically determine based on directory content
                test_paths = OrderedDict({
                        'gray'   : _glob_dir(args.test_dir, glob_strs['gray']),
                        'white'  : _glob_dir(args.test_dir, glob_strs['white']),
                        'back'   : _glob_dir(args.test_dir, glob_strs['back']),
                        'tissue' : _glob_dir(args.test_dir, glob_strs['tissue']) 
                        })

                # Remove empty lists
                test_paths = OrderedDict({k: v for k, v in test_paths.items() if v})

                assert test_paths, 'No recognized mask file ' \
                        f'{list(glob_strs.values())} '\
                        f'found in {args.test_dir}'
            # Check we have all masks for the same number of images
            _assert_same_len_values_in_dict(test_paths)
            return test_paths

        def _get_image_names(test_paths: Dict[str, List[str]]) -> List[str]:
            mask, mask_paths = next(iter(test_paths.items()))

            return [p.split('/')[-1].split(glob_strs[mask][1:])[0] 
                    for p in mask_paths]

        def _print_config(image_names: List[str], metrics: List[str]) -> None:
            print(f'\nCompute mask accuracy for {len(image_names)} WSIs : '\
                    f'{[k for k in test_paths.keys()]}' )
            print(f'\tusing {metrics} metrics\n')

        def _get_accuracy_metrics() -> List[str]:
            if any([args.pixel_accuracy, args.iou, args.f1_score]):
                metrics = []
                if args.pixel_accuracy:
                    metrics.append('Pixel_Accuracy')
                if args.mean_accuracy:
                    metrics.append('Mean_Accuracy')
                if args.iou:
                    metrics.append('IoU')
                if args.mean_iou:
                    metrics.append('Mean_IoU')
                if args.frequency_iou:
                    metrics.append('Frequency_Weighted_IoU')
                if args.f1_score:
                    metrics.append('F1_Score')
            else:   # by default, use everything except F1 score
                metrics = ['Pixel_Accuracy', 'Mean_Accuracy', 'IoU', 
                        'Mean_IoU', 'Frequency_Weighted_IoU']
            return metrics

        def _compute_confusion_matrix(truth_path: str, test_path: str) -> "OrderedDict[str, Number]":
            truth_img = Image.open(truth_path)
            test_img  = Image.open(test_path)

            # If two sizes mismatch, upsample test_img to fit truth_img
            if test_img.size != truth_img.size:
                print(f'\tsize mismatch, upsampling "{test_path}" '\
                        f'from {test_img.size} to {truth_img.size}')
                test_img = test_img.resize(truth_img.size)

            truth_img_arr = np.array(truth_img)
            test_img_arr  = np.array(test_img)
            assert truth_img_arr.dtype == np.bool, 'Wrong groundtruth image type, '\
                    f'expected "np.bool" but got {truth_img_arr.dtype}'
            assert test_img_arr.dtype == np.bool, 'Wrong testing image type, '\
                    f'expected "np.bool" but got {test_img_arr.dtype}'
            del truth_img, test_img

            NOT_truth_img_arr = np.logical_not(truth_img_arr)
            NOT_test_img_arr  = np.logical_not(test_img_arr)

            TP = np.multiply(test_img_arr, truth_img_arr).sum()
            FP = np.multiply(test_img_arr, NOT_truth_img_arr).sum()
            FN = np.multiply(NOT_test_img_arr, truth_img_arr).sum()
            TN = np.multiply(NOT_test_img_arr, NOT_truth_img_arr).sum()

            del truth_img_arr, test_img_arr, NOT_truth_img_arr, NOT_test_img_arr

            return OrderedDict({'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN})

        def _compute_metrics(conf_dict: "OrderedDict[str, Number]", metrics: List[str]) -> None:
            total = conf_dict['TP'] + conf_dict['FP'] + \
                    conf_dict['FN'] + conf_dict['TN']

            P = conf_dict['TP'] + conf_dict['FN']

            # Common metrics
            conf_dict['Accuracy (%)'] = \
                    (conf_dict['TP'] + conf_dict['TN']) / total * 100
            conf_dict['Misclassification_Rate (%)'] = \
                    (conf_dict['FP'] + conf_dict['FN']) / total * 100
            conf_dict['Sensitivity (%)'] = \
                    conf_dict['TP'] / (conf_dict['TP'] + conf_dict['FN']) * 100
            conf_dict['Specificity (%)'] = \
                    conf_dict['TN'] / (conf_dict['TN'] + conf_dict['FP']) * 100
            conf_dict['Precision (%)'] = \
                    conf_dict['TP'] / (conf_dict['TP'] + conf_dict['FP']) * 100
            conf_dict['Prevalence (%)'] = \
                    (conf_dict['FN'] + conf_dict['TP']) / total * 100

            # Specified metrics
            for metric in metrics:
                if metric == 'IoU':
                    conf_dict['IoU (%)'] = conf_dict['TP'] / \
                            (conf_dict['TP'] + conf_dict['FP'] + conf_dict['FN']) * 100
                elif metric == 'F1_Score':
                    conf_dict['F1_Score (%)'] = 2 * conf_dict['TP'] / \
                            (2 * conf_dict['TP'] + conf_dict['FP'] + conf_dict['FN']) * 100
                elif metric == 'Pixel_Accuracy':
                    conf_dict['Pixel_Accuracy (%)'] = \
                            conf_dict['TP'] / total * 100
                elif metric == 'Mean_Accuracy':
                    conf_dict['Mean_Accuracy (%)'] = \
                            conf_dict['TP'] / P * 100
                elif metric == 'Mean_IoU':
                    conf_dict['Mean_IoU (%)'] = conf_dict['TP'] / \
                            (conf_dict['TP'] + conf_dict['FP'] + conf_dict['FN']) * 100
                elif metric == 'Frequency_Weighted_IoU':
                    conf_dict['Frequency_Weighted_IoU (%)'] = P / total * \
                            conf_dict['TP'] / \
                            (conf_dict['TP'] + conf_dict['FP'] + conf_dict['FN']) * 100

        def _get_mean(in_list: List[Number]) -> float:
            return sum(in_list) / len(in_list)


        # Glob strings for mask .png files
        glob_strs = {
                'gray'   : "*-Gray.png",
                'white'  : "*-White.png",
                'back'   : "*-Background.png",
                'tissue' : "*-Tissue.png"
                }

        # Get test image paths
        test_paths = _get_test_image_paths()

        # Select accuracy metric
        metrics = _get_accuracy_metrics()

        image_names = _get_image_names(test_paths)
        _print_config(image_names, metrics)


        ##### Begin calculation #####
        # total_results is a List of Dictionary
        total_results = []
        for i, image_name in enumerate(image_names):
            image_results: OrderedDict[str, Any] = OrderedDict({'Image Name': image_name})
            print('[%3d/%3d]\tEvaluating masks of %s' 
                    % (i+1, len(image_names), image_name))

            for mi, mask_name in enumerate(test_paths.keys()):
                if mask_name == 'tissue':
                    truth_img_path = os.path.join(args.truth_dir[0], 
                            image_name+glob_strs['back'][1:])
                    for truth_dir in args.truth_dir[1:]:
                        if not os.path.exists(truth_img_path):
                            truth_img_path = os.path.join(truth_dir, 
                                    image_name+glob_strs['back'][1:])
                else:
                    truth_img_path = os.path.join(args.truth_dir[0], 
                            image_name+glob_strs[mask_name][1:])
                    for truth_dir in args.truth_dir[1:]:
                        if not os.path.exists(truth_img_path):
                            truth_img_path = os.path.join(truth_dir, 
                                    image_name+glob_strs[mask_name][1:])
                test_img_path = os.path.join(args.test_dir, 
                        image_name+glob_strs[mask_name][1:])

                conf_dict = \
                        _compute_confusion_matrix(truth_img_path, test_img_path) # type: ignore

                # Modify confusion matrix if evaluating tissue masks
                if mask_name == 'tissue':
                    tissue_conf_dict = OrderedDict({
                            'TP' : conf_dict['FP'],
                            'FP' : conf_dict['TP'],
                            'FN' : conf_dict['TN'],
                            'TN' : conf_dict['FN'] })
                    conf_dict = tissue_conf_dict

                # Compute accuracy metrics
                _compute_metrics(conf_dict, metrics)

                # Add conf_dict to image_results
                for k, v in conf_dict.items():
                    # Accumulate these accuracy metrics
                    if k.startswith('Pixel_Accuracy') \
                            or k.startswith('Frequency_Weighted_IoU'):
                        image_results[k] = image_results[k] + v \
                                if k in image_results else v

                    # Accumulate these accuracy metrics 
                    # and divided by n_class at the end
                    elif k.startswith('Mean_Accuracy') \
                            or k.startswith('Mean_IoU'):
                        image_results[k] = image_results[k] + v \
                                if k in image_results else v
                        if mi == len(test_paths.keys()) - 1:
                            image_results[k] /= len(test_paths.keys())
                    # Append mask_name as a prefix of conf_dict.keys()
                    else:
                        image_results[str(mask_name.capitalize()+'-'+k)] = v

                # Move these metrics to the end
                for k, v in conf_dict.items():
                    if k.startswith('Pixel_Accuracy') \
                            or k.startswith('Frequency_Weighted_IoU') \
                            or k.startswith('Mean_Accuracy') \
                            or k.startswith('Mean_IoU'):
                        image_results.move_to_end(k)

            total_results.append(image_results)


        ##### Generate csv and LaTeX table tex #####
        output_dir = os.path.join(args.test_dir, 'results')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        cur_time = datetime.now()
        csv_file_path = os.path.join(output_dir, 
                cur_time.strftime("%Y%m%d_%H%M%S.csv"))
        tex_file_path = os.path.join(output_dir, 
                cur_time.strftime("%Y%m%d_%H%M%S.tex"))

        with open(csv_file_path, "w") as f:
            headers = list(total_results[0].keys())
            f.write(','.join(headers) + '\n')

            # Write out accuracy results for each image
            for i, image_results in enumerate(total_results):
                for hi, header in enumerate(headers):
                    if header == 'Image Name':
                        f.write(image_results[header])
                    elif '%' in header:
                        f.write('%.2f' % (image_results[header]))
                    else:
                        f.write('%d' % (image_results[header]))

                    # Write separator or new line
                    if hi != len(headers) - 1:
                        f.write(',')
                    else:
                        f.write('\n')

            # Write out average results
            for hi, header in enumerate(headers):
                if header == 'Image Name':
                    f.write('Average')
                elif '%' in header:
                    f.write('%.2f' % 
                            (_get_mean([d[header] for d in total_results])) )
                else:
                    f.write('%d' % 
                            (_get_mean([d[header] for d in total_results])) )

                # Write separator or new line
                if hi != len(headers) - 1:
                    f.write(',')
                else:
                    f.write('\n')

        with open(tex_file_path, "w") as f:
            # List of headers to print
            headers = ['Image Name']
            headers += [k for k in list(total_results[0].keys()) 
                    for m in metrics if m in k]
            headers = list(dict.fromkeys(headers))  # remove duplicates

            tabular_str = '||' + 'c|' * len(headers) + '|'

            # Write table prologue
            f.write(dedent(r'''
                    \begin{table}[H]
                        \centering
                        \setlength{\belowcaptionskip}{-0.cm}
                        \begin{tabular}{''') + tabular_str + '}\n')

            # Write the headers
            f.write(' '*8 + r'\hline' + '\n' + \
                    ' '*8 + 'WSI Name & ' + \
                    ' & '.join([f"{s.replace('_', '-').replace(' (%)', '')}"
                        for s in headers[1:]]) + r' \\' + '\n' + \
                    ' '*8 + r'\hline'*2 + '\n')

            # Write the accuracy results
            for i, image_results in enumerate(total_results):
                for hi, header in enumerate(headers):
                    if header == 'Image Name':
                        f.write(' '*8 + image_results[header][:6])
                    elif '%' in header:
                        f.write('%.2f' % (image_results[header]) + r'\%')

                    # Write separator or new line
                    if hi != len(headers) - 1:
                        f.write(' & ')
                    elif i != len(total_results) - 1:
                        f.write(r' \\' + '\n' + ' '*8 + r'\hline' + '\n')
                if i == len(total_results) - 1:
                    f.write(r' \\' + '\n' + ' '*8 + r'\hline'*2 + '\n')

            # Write out average results
            for hi, header in enumerate(headers):
                if header == 'Image Name':
                    f.write(' '*8 + 'Average')
                elif '%' in header:
                    f.write('%.2f' % 
                            (_get_mean([d[header] for d in total_results])) + r'\%')

                # Write separator or new line
                if hi != len(headers) - 1:
                    f.write(' & ')
                else:
                    f.write(r' \\' + '\n' + ' '*8 + r'\hline' + '\n')

            # Write table epilogue
            f.write(dedent(r'''
                        \end{tabular}
                        \caption{Insert caption here}
                        \label{tab:accuracy}
                    \end{table}
                    '''))

        print('\nFinished computing mask accuracy\n' + \
                f'\tA csv result is saved as "{csv_file_path}"\n' + \
                f'\tA LaTeX table is saved as "{tex_file_path}"\n')


if __name__ == '__main__':
    ### For testing ###
    parser = ComputeMaskAccuracy.get_parser()
    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    ComputeMaskAccuracy(args)

    print("Done!")
