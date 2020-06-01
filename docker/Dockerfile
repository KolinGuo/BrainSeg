FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04
#FROM tensorflow/tensorflow:nightly-gpu-jupyter
#FROM tensorflow/tensorflow:2.1.0-gpu-py3-jupyter

#########################################
# SECTION 1: Essentials                 #
#########################################
RUN apt-get update && apt-get install -y --no-install-recommends \
    vim git curl wget yasm cmake unzip pkg-config \
    checkinstall build-essential ca-certificates \
    software-properties-common apt-utils bash-completion \
  && apt-get upgrade -y \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

#########################################
# SECTION 2: Setup Image Libraries      #
#########################################
RUN apt-get update && apt-get install -y --no-install-recommends \
    zlib1g-dev libjpeg-dev libpng-dev xvfb ffmpeg xorg-dev \
    xorg-dev libboost-all-dev libsdl2-dev swig \
    libblas-dev liblapack-dev \
    libopenblas-base libatlas-base-dev graphviz \
    libvips \
  && apt-get upgrade -y \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# libffi-dev: interface that allows code written in one language to call code written in another language.
#RUN apt-get update && apt-get install -y --no-install-recommends \
#  libssl-dev libffi-dev \
#  libatlas-base-dev gfortran \
#  python-software-properties \
#  && rm -rf /var/lib/apt/lists/* \
#  && apt upgrade -y

#########################################
# SECTION 3: Install Python Libraries   #
#########################################
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install Python libraries
COPY install/requirements.txt /tmp/
RUN pip3 install -U -r /tmp/requirements.txt

# Install jupyter notebook extensions (code folding)
RUN pip3 install jupyter_contrib_nbextensions \
  && jupyter contrib nbextension install \
  && pip3 install jupyter_nbextensions_configurator \
  && jupyter nbextensions_configurator enable

# Jupyter runtime folder
RUN mkdir /.local && chmod a+rwx /.local

# Install argcomplete for auto tab completing argparse in Python
RUN pip3 install argcomplete \
  && activate-global-python-argcomplete

######################################
# SECTION 4: OpenCV packages         #
######################################
ARG OPENCV_PATH=/usr/local/share/opencv
ARG OPENCV_CONTRIB_PATH=/usr/local/share/opencv_contrib
ARG OPENCV_VERSION=4.3.0

RUN apt-get update && apt-get -y install \
    libglew-dev \
    libjpeg-dev libpng-dev libtiff-dev \
    libavcodec-dev libavformat-dev libswscale-dev libavresample-dev \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    libxvidcore-dev x264 libx264-dev libfaac-dev libmp3lame-dev libtheora-dev \
    libopencore-amrnb-dev libopencore-amrwb-dev \
    libdc1394-22 libdc1394-22-dev libxine2-dev libv4l-dev v4l-utils \
    libgtk-3-dev \
    python3-dev python3-testresources \
    libtbb-dev \
    libatlas-base-dev gfortran \
    libprotobuf-dev protobuf-compiler \
    libgoogle-glog-dev libgflags-dev \
    libgphoto2-dev libeigen3-dev libhdf5-dev doxygen \
    libegl1-mesa-dev \
    libwayland-dev \
    libxkbcommon-dev \
    wayland-protocols

RUN echo -e "Install OPENCV Contrib\n\n" && \
    wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/${OPENCV_VERSION}.zip && \
    unzip opencv_contrib.zip && \
    mv opencv_contrib-* $OPENCV_CONTRIB_PATH && \
    rm opencv_contrib.zip

RUN echo -e "Install OPENCV \n\n" && \
    wget -O opencv.zip https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip && \
    unzip opencv.zip && \
    mv opencv-* $OPENCV_PATH && \
    rm opencv.zip

RUN echo -e "Make / Build OPENCV\n\n" && \
    mkdir -p $OPENCV_PATH/build && \
    cd $OPENCV_PATH/build && \
    cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=${OPENCV_PATH} \
    -D OPENCV_EXTRA_MODULES_PATH=${OPENCV_CONTRIB_PATH}/modules \
    -D WITH_CUDA=ON \
    -D CUDA_ARCH_BIN=6.1 \
    -D CUDA_ARCH_PTX="" \
    -D WITH_CUDNN=ON \
    -D WITH_CUBLAS=ON \
    -D CUDA_FAST_MATH=ON \
    -D ENABLE_FAST_MATH=ON \
    -D WITH_TBB=ON \
    -D WITH_V4L=ON \
    -D WITH_QT=OFF \
    -D WITH_OPENGL=ON \
    -D BUILD_opencv_cudacodec=ON \
    -D INSTALL_C_EXAMPLES=OFF \
    -D INSTALL_PYTHON_EXAMPLES=OFF \
    -D BUILD_opencv_python2=OFF \
    -D BUILD_NEW_PYTHON_SUPPORT=ON \
    -D PYTHON3_EXECUTABLE=$(which python3) \
    -D PYTHON3_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
    -D OPENCV_PYTHON3_INSTALL_PATH=$(python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") \
    -D BUILD_EXAMPLES=OFF \
    .. && \
    make -j$(nproc) && make install

ENV OpenCV_DIR=/usr/local/share/opencv/build

######################################
# SECTION 5: Add running instruction #
######################################
# Set working directory to be repository directory
ENV REPOPATH /BrainSeg
WORKDIR ${REPOPATH}

# Append the custom bashrc
COPY bashrc /tmp/
RUN cat /tmp/bashrc > /etc/bash.bashrc

# For Tensorboard profiler
ENV LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:${LD_LIBRARY_PATH}

# Setup locale language config
RUN apt-get update && apt-get -y install locales && \
    locale-gen "en_US.UTF-8" && \
    update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8
