#!/bin/bash 
# Ensure that you have installed docker(API >= 1.40) and the nvidia graphics driver on host!

############################################################
# Section 0: Project-Specific Settings                     #
############################################################
IMGNAME="brainseg"
CONTNAME="brainseg"
DOCKERFILEPATH="./docker/Dockerfile"
REPONAME="BrainSeg"
JUPYTERPORT="9000"
TENSORBOARDPORT="6006"

COMMANDTOINSTALL="cd /BrainSeg/src/gSLICr && rm -rf build && mkdir build && cd build && cmake .. && make -j$(nproc) && cd /BrainSeg"
COMMANDTORUN="jupyter notebook --no-browser --ip=0.0.0.0 --allow-root --port=$JUPYTERPORT &"
COMMANDTOSTARTCONTAINER="docker start -ai $CONTNAME"

############################################################
# Section 1: Helper Function Definition                    #
############################################################
SCRIPT=$(readlink -f "$0")
SCRIPTPATH=$(dirname "$SCRIPT")
cd "$SCRIPTPATH"

USAGE="Usage: ./setup.sh [rmimcont=[0,1]] [rmimg=[0,1]]\n"
USAGE+="\trmimcont=[0,1] : 0 to not remove intermediate Docker containers\n"
USAGE+="\t                 after a successful build and 1 otherwise\n"
USAGE+="\t                 default is 1\n"
USAGE+="\trmimg=[0,1]    : 0 to not remove previously built Docker image\n"
USAGE+="\t                 and 1 otherwise\n"
USAGE+="\t                 default is 0\n"

REMOVEIMDDOCKERCONTAINERCMD="--rm=true"
REMOVEPREVDOCKERIMAGE=false

test_retval() {
  if [ $? -ne 0 ] ; then
    echo -e "\nFailed to ${*}... Exiting...\n"
    exit 1
  fi
}

parse_argument() {
  # Parsing argument
  if [ $# -ne 0 ] ; then
    while [ ! -z $1 ] ; do
      if [ "$1" = "rmimcont=0" ] ; then
        REMOVEIMDDOCKERCONTAINERCMD="--rm=false"
      elif [ "$1" = "rmimg=1" ] ; then
        REMOVEPREVDOCKERIMAGE=true
      elif [[ "$1" != "rmimcont=1" && "$1" != "rmimg=0" ]] ; then
        echo -e "Unknown argument: " $1
        echo -e "$USAGE"
        exit 1
      fi
      shift
    done
  fi
}

print_setup_info() {
  # Echo the set up information
  echo -e "\n\n"
  echo -e "################################################################################\n"
  echo -e "\tSet Up Information\n"
  if [ "$REMOVEIMDDOCKERCONTAINERCMD" = "--rm=true" ] ; then
    echo -e "\t\tRemove intermediate Docker containers after a successful build\n"
  else
    echo -e "\t\tKeep intermediate Docker containers after a successful build\n"
  fi
  if [ "$REMOVEPREVDOCKERIMAGE" = true ] ; then
    echo -e "\t\tCautious!! Remove previously built Docker image\n"
  else
    echo -e "\t\tKeep previously built Docker image\n"
  fi
  echo -e "################################################################################\n"
}

remove_prev_docker_image () {
  # Remove previously built Docker image
  if [ "$REMOVEPREVDOCKERIMAGE" = true ] ; then
    echo -e "\nRemoving previously built image..."
    docker rmi -f $IMGNAME
  fi
}

create_custom_bashrc() {
  echo > bashrc

  # Change PS1, terminal color, and cmd colors
  echo export PS1=\"\\[\\e[31m\\]${CONTNAME}-docker\\[\\e[m\\] \\[\\e[33m\\]\\w\\[\\e[m\\] \> \" >> bashrc \
    && echo export TERM=xterm-256color >> bashrc \
    && echo alias grep=\"grep --color=auto\" >> bashrc \
    && echo alias ls=\"ls --color=auto\" >> bashrc \
    && echo >> bashrc

  # Change terminal color and boldness
  echo "# Cyan color" >> bashrc \
    && echo echo -e \"\\e[1\;36m\" >> bashrc \
    && echo >> bashrc

  # If there is an installing command inside docker container
  if [ ! -z "$COMMANDTOINSTALL" ] ; then
    echo -n COMMANDTOINSTALL=\" >> bashrc \
      && echo -n ${COMMANDTOINSTALL} | sed 's/\"/\\"/g' >> bashrc \
      && echo \" >> bashrc \
      && echo >> bashrc
  fi

  # Echo command to run the application
  if [ ! -z "$COMMANDTORUN" ] ; then
    echo -n COMMANDTORUN=\" >> bashrc \
      && echo -n ${COMMANDTORUN} | sed 's/\"/\\"/g' >> bashrc \
      && echo \" >> bashrc \
      && echo >> bashrc
  fi

  # Echo the echo command to print instructions
  echo echo -e \"\" >> bashrc
  echo echo -e \"################################################################################\\n\" >> bashrc
  if [ ! -z "$COMMANDTOINSTALL" ] ; then
    echo echo -e \"\\tCommand to install ${REPONAME} for the first time:\\n\\t\\t'${COMMANDTOINSTALL}'\\n\" >> bashrc
  fi
  if [ ! -z "$COMMANDTORUN" ] ; then
    echo echo -e \"\\tCommand to run:\\n\\t\\t'${COMMANDTORUN}'\\n\" >> bashrc
  fi
  echo echo -e \"################################################################################\\n\" >> bashrc \
    && echo >> bashrc

  # Change terminal color back
  echo "# Turn off colors" >> bashrc \
    && echo echo -e \"\\e[m\" >> bashrc
}

build_docker_image() {
  # Set REPOPATH for WORKDIR
  sed -i "s/^ENV REPOPATH.*$/ENV REPOPATH \/${REPONAME}/" $DOCKERFILEPATH

  # Build and run the image
  echo -e "\nBuilding image $IMGNAME..."
  docker build $REMOVEIMDDOCKERCONTAINERCMD -f $DOCKERFILEPATH -t $IMGNAME .
  test_retval "build Docker image $IMGNAME"
  rm -rf bashrc
}

build_docker_container() {
  # Build a container from the image
  echo -e "\nRemoving older container $CONTNAME..."
  if [ 1 -eq $(docker container ls -a | grep "$CONTNAME$" | wc -l) ] ; then
    docker rm -f $CONTNAME
  fi

  echo -e "\nBuilding a container $CONTNAME from the image $IMGNAME..."
  docker create -it --name=$CONTNAME \
    -u $(id -u):$(id -g) \
    -v "$SCRIPTPATH":/$REPONAME \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /etc/localtime:/etc/localtime:ro \
    -e DISPLAY=$DISPLAY \
    --ipc=host \
    --gpus all \
    -p $JUPYTERPORT:$JUPYTERPORT \
    -p $TENSORBOARDPORT:$TENSORBOARDPORT \
    --privileged=true \
    $IMGNAME /bin/bash
  test_retval "create Docker container"
}

start_docker_container() {
  docker start -ai $CONTNAME
  
  if [ 0 -eq $(docker container ls -a | grep "$CONTNAME$" | wc -l) ] ; then
    echo -e "\nFailed to start/attach Docker container... Exiting...\n"
    exit 1
  fi
}

print_exit_command() {
  # Echo command to start container
  echo -e "\n"
  echo -e "################################################################################\n"
  echo -e "\tCommand to start Docker container:\n\t\t${COMMANDTOSTARTCONTAINER}\n"
  echo -e "################################################################################\n"
}


############################################################
# Section 2: Call Helper Functions                         #
############################################################
# Parse shell script's input arguments
parse_argument "$@"
# Print the setup info
print_setup_info
# Print usage of the script
echo -e "\n$USAGE\n"

echo -e ".......... Set up will start in 5 seconds .........."
sleep 5

remove_prev_docker_image
create_custom_bashrc
build_docker_image
build_docker_container
start_docker_container

# When exit from docker container
print_exit_command
