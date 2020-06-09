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

COMMANDTOINSTALLGSLICR="cd /${REPONAME}/src/gSLICr && rm -rf build && mkdir build && cd build && cmake .. && make -j$(nproc) && cd /BrainSeg"
COMMANDTORUNTENSORBOARD="tensorboard --logdir /${REPONAME}/tf_logs/ --port ${TENSORBOARDPORT} --host 0.0.0.0 >/dev/null 2>&1 &"
COMMANDTOSTARTCONTAINER="docker start -ai ${CONTNAME}"

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
  cat > bashrc <<- "EOF"
# If not running interactively, don't do anything
[ -z "$PS1" ] && return

# check the window size after each command and, if necessary,
# update the values of LINES and COLUMNS.
shopt -s checkwinsize

# enable bash completion in interactive shells
if ! shopt -oq posix; then
  if [ -f /usr/share/bash-completion/bash_completion ]; then
    . /usr/share/bash-completion/bash_completion
  elif [ -f /etc/bash_completion ]; then
    . /etc/bash_completion
  fi
fi

. /etc/bash_completion

# if the command-not-found package is installed, use it
if [ -x /usr/lib/command-not-found -o -x /usr/share/command-not-found/command-not-found ]; then
  function command_not_found_handle {
    # check because c-n-f could've been removed in the meantime
    if [ -x /usr/lib/command-not-found ]; then
      /usr/lib/command-not-found -- "$1"
      return $?
    elif [ -x /usr/share/command-not-found/command-not-found ]; then
      /usr/share/command-not-found/command-not-found -- "$1"
      return $?
    else
      printf "%s: command not found\n" "$1" >&2
      return 127
    fi
  }
fi

# Change PS1 and terminal color
export PS1="\[\e[31m\]brainseg-docker\[\e[m\] \[\e[33m\]\w\[\e[m\] > "
export TERM=xterm-256color
alias grep="grep --color=auto"
alias ls="ls --color=auto"

# Cyan color
echo -e "\e[1;36m"

EOF

  # If there is an installing command inside docker container
  if [ ! -z "$COMMANDTOINSTALLGSLICR" ] ; then
    echo -n COMMANDTOINSTALLGSLICR=\" >> bashrc \
      && echo -n ${COMMANDTOINSTALLGSLICR} | sed 's/\"/\\"/g' >> bashrc \
      && echo \" >> bashrc \
      && echo >> bashrc
  fi

  # Echo command to run the application
  if [ ! -z "$COMMANDTORUNTENSORBOARD" ] ; then
    echo -n COMMANDTORUNTENSORBOARD=\" >> bashrc \
      && echo -n ${COMMANDTORUNTENSORBOARD} | sed 's/\"/\\"/g' >> bashrc \
      && echo \" >> bashrc \
      && echo >> bashrc
  fi

  # Echo the echo command to print instructions
  echo echo -e \"\" >> bashrc
  echo echo -e \"################################################################################\\n\" >> bashrc
  if [ ! -z "$COMMANDTOINSTALLGSLICR" ] ; then
    echo echo -e \"\\tCommand to install gSLICr for the first time:\\n\\t\\t'${COMMANDTOINSTALLGSLICR}'\\n\" >> bashrc
  fi
  if [ ! -z "$COMMANDTORUNTENSORBOARD" ] ; then
    echo echo -e \"\\tCommand to run:\\n\\t\\t'${COMMANDTORUNTENSORBOARD}'\\n\" >> bashrc
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
