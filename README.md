# BrainSeg
Automated Grey and White Matter Segmentation in Digitized A*β*
Human Brain Tissue WSI

## Authors / Contributors
* Runlin Guo
* Wenda Xu
* Zhengfeng Lai

If you have any questions/suggestions or find any bugs,
please submit a GitHub issue.

## Prerequisites
The list of prerequisites for building and running this repository is described
below. It is guaranteed to run out-of-the-box on the configuration below.
Otherwise, some trial-and-error processes might involve.
* Ubuntu 16.04/18.04 on AMD64 CPU
* System RAM >= 128 GB (less might be okay depending on usage)
* NVIDIA GPU with CUDA version >= 10.1
* [Docker](https://docs.docker.com/install/linux/docker-ce/ubuntu/)
version >= 19.03, API >= 1.40
* [nvidia-container-toolkit](https://github.com/NVIDIA/nvidia-docker#ubuntu-16041804-debian-jessiestretchbuster)
(previously known as nvidia-docker)  

Command to test if all prerequisites are met:  
`sudo docker run -it --rm --gpus all ubuntu nvidia-smi`

## Setup Instructions
`bash ./setup.sh`  
You should be greeted by the Docker container **brainseg** when this script
finishes. The working directory is */BrainSeg* which is where the repo is
mounted at.  

#### Port allocations
* Jupyter notebook: 9000
* TensorBoard: 6006

## Repo Directory Structure
* [docker/](docker) - Dockerfiles
* [install/](install) - Python third-party libraries requirements
* [notebook/](notebook) - Jupyter notebooks including those from
[plaquebox-paper (original repo)](https://github.com/keiserlab/plaquebox-paper).
Some special ones are:
    * [Grad-CAM.ipynb](notebook/Grad-CAM.ipynb) - Grad-CAM visualization for
      TensorFlow2.
    * [Mask_Accuracy_Benchmark_Plotting.ipynb](notebook/Mask_Accuracy_Benchmark_Plotting.ipynb) -
      Mask accuracy plotting for nice-looking graphs using *.csv* files from
      [compute_mask_accuracy.py](src/utils/compute_mask_accuracy.py).
    * [PowerAnalysis.ipynb](notebook/PowerAnalysis.ipynb) - Power analysis.
    * [TensorBoard_Plotting.ipynb](notebook/TensorBoard_Plotting.ipynb) -
      Replot TensorBoard data to nice-looking graphs.
* [qupath/](qupath) - QuPath tracing scripts for generating groundtruth masks.
  Brief tutorial can be found [here](https://docs.google.com/document/d/125n8o4KQlUcEIbycHDTXV-8pBcj-CLsxISnygt0SecM/edit?usp=sharing).
* [src/](src) - Source code
    * [gSLICr/](src/gSLICr) -
      [gSLICr (GPU-based SLIC implementation)](https://github.com/carlren/gSLICr)
      repo with added support for
      [SLICO](https://www.epfl.ch/labs/ivrl/research/slic-superpixels/#SLICO)
      and CUDA unified memory. Building instruction can be found in
      [setup.sh](setup.sh#L14).
    * [networks/](src/networks) - Network related code
        * [models/](src/networks/models) - various network model implementations
            * [FCN.py](src/networks/models/FCN.py) - FCN Model
            * [UNet.py](src/networks/models/UNet.py) - UNet Model
            * [models.py](src/networks/models/models.py) - top-level model selection
        * [dataset.py](src/networks/dataset.py) - Dataset related functions/classes.
        * [losses.py](src/networks/losses.py) - Custom losses based on `tf.keras.losses`.
        * [metrics.py](src/networks/metrics.py) - Custom metrics based on `tf.keras.metrics`.
    * [utils/](src/utils) - Utility functions
        * [color_deconv.py](src/utils/color_deconv.py) - Color deconvolution
          (color space transformation) from RGB space to stain color space.
        * [compute_mask_accuracy.py](src/utils/compute_mask_accuracy.py) -
          Evaluate mask accuracy using various metrics. Implemented metrics can
          be found via `./compute_mask_accuracy.py -h`
        * [numpy_pil_helper.py](src/utils/numpy_pil_helper.py) -
          Conversion functions between NumPy and PIL.
        * [separate_tissue.py](src/utils/separate_tissue.py) -
          Separates tissue from background using traditional image processing
          techniques with gSLICr.
        * [svs_to_png.py](src/utils/svs_to_png.py) -
          Convert *.svs* WSI to *.png* images via tiling.
    * [image_helper.py](src/image_helper.py) - Image helper scripts including
      commonly used functions like `grayscale_to_binary`, `get_thumbnails`,
      `combine_truth_binary`, and `resize_to_original`.
    * [postproc.py](src/postproc.py) - Post-processing script
    * [predict.py](src/predict.py) - Predicting script for trained network inferencing
    * [tissue_seg.py](src/tissue_seg.py) - Tissue separation script which calls
      [utils/separate_tissue.py](src/utils/separate_tissue.py)
      iteratively over a directory of *.svs* WSI image.
      **Note: manual stain estimation using QuPath is required.**
    * [train.py](src/train.py) - Training script
* [tests/](tests) - Testing cases for source code using Python `unittest`.

#### Note
In this repo, all Python scripts that are meant to be run as top-level scripts
support [`argparse`](https://docs.python.org/3/library/argparse.html).  
That is, input arguments can be specified to change the default behavior.
List of input arguments and their usage instructions can be found by
`./my_python_script.py -h` or `./my_python_script.py <subcommand> -h`.

## Credit
* [plaquebox-paper (original repo)](https://github.com/keiserlab/plaquebox-paper)
* [plaquebox-paper (fork with some code dev)](https://github.com/KolinGuo/plaquebox-paper)
* [QuPath](https://qupath.github.io/)
* [pyvips](https://libvips.github.io/pyvips/): Image processing for large
TIFF-based image.
* [gSLICr (GPU-based SLIC implementation)](https://github.com/carlren/gSLICr):
