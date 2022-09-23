FROM tensorflow/tensorflow:2.6.1-gpu

ENV TZ=Asia/Seoul
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list

# Install sudo
RUN apt-get update && \
    apt-get install -y sudo apt-utils curl

# Environment config
ENV DEBIAN_FRONTEND=noninteractive

# Add new sudo user
ARG user=ros
ARG passwd=ros
ARG uid=1000
ARG gid=1000
ENV USER=$user
ENV PASSWD=$passwd
ENV UID=$uid
ENV GID=$gid
RUN useradd --create-home -m $USER && \
        echo "$USER:$PASSWD" | chpasswd && \
        usermod --shell /bin/bash $USER && \
        usermod -aG sudo $USER && \
        echo "$USER ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/$USER && \
        chmod 0440 /etc/sudoers.d/$USER && \
        # Replace 1000 with your user/group id
        usermod  --uid $UID $USER && \
        groupmod --gid $GID $USER

# Install other utilities
RUN apt-get update && \
    apt-get install -y vim \
    tmux \
    git \
    wget \
    lsb-release \
    lsb-core \
    python3-pip 


### ROS INSTALLATION ###
# setup sources.list
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu `lsb_release -cs` main" > /etc/apt/sources.list.d/ros-latest.list' && \
    curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add - && \
    apt-get update && apt-get install -y ros-noetic-ros-base
    
RUN apt-get install -y python3-rosinstall \
    python3-rosdep \
    python3-rosinstall \
    python3-rosinstall-generator \
    python3-wstool  \ 
    build-essential 

RUN rosdep init && \
    rosdep update


# setup keys
#RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
#    apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key 421C365BD9FF1F717815A3895523BAEEB01FA116 && \


# Setup ROS
USER $USER
RUN rosdep fix-permissions && rosdep update
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
RUN /bin/bash -c "source ~/.bashrc"

WORKDIR /home/$USER
RUN mkdir -p ~/catkin_ws/src && \
    cd ~/catkin_ws && \
    . /opt/ros/noetic/setup.sh && \
    catkin_make

# Pull git-hub packages
WORKDIR /home/$USER
RUN mkdir packages && \
    cd packages && \
    git clone https://github.com/JureHudoklin/trimesh_visualize.git && \
    cd trimesh_visualize && \
    pip install -r requirements.txt && \
    pip install -e .

WORKDIR /home/$USER
RUN cd ~/catkin_ws/src && \
    git clone --recurse https://github.com/JureHudoklin/ContinuosSuctionCupModel.git && \
    cd ContinuosSuctionCupModel && \
    pip install -r requirements.txt && \
    cd ~/catkin_ws && \
    . /opt/ros/noetic/setup.sh && \
    catkin_make

WORKDIR /home/$USER/catkin_ws/src/ContinuosSuctionCupModel/pointnet2-tensorflow2
RUN chmod u+x tf_ops/compile_ops.sh && \
    tf_ops/compile_ops.sh

# Expose Jupyter 
EXPOSE 8888

# Expose Tensorboard
EXPOSE 6006