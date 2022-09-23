FROM tensorflow/tensorflow:2.3.1-gpu

ENV TZ=Asia/Seoul
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

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


# Setup ROS
USER $USER

WORKDIR /home/$USER
RUN mkdir -p ~/catkin_ws/src && \
    cd ~/catkin_ws && \
    catkin_make && \
    echo "source ~/catkin_ws/devel/setup.bash" >> ~/.bashrc

# Pull git-hub packages
WORKDIR /home/$USER
RUN mkdir packages && \
    cd packages && \
    git clone https://github.com/JureHudoklin/trimesh_visualize.git && \
    cd trimesh_visualize && \
    pip3 install -r requirements.txt && \
    pip install -e .

WORKDIR /home/$USER
RUN cd ~/catkin_ws/src && \
    git clone https://github.com/JureHudoklin/ContinuosSuctionCupModel.git && \
    cd ~/catkin_ws && \
    catkin_make


# Expose Jupyter 
EXPOSE 8888

# Expose Tensorboard
EXPOSE 6006