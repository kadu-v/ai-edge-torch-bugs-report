FROM ubuntu:22.04
ENV DEBIAN_FRONTEND noninteractive
ENV LANG C.UTF-8
ENV TZ Asia/Tokyo

# Install some common tools for ubuntu
RUN apt-get update \
    && apt-get install -y \
    sudo \
    xxd \
    wget \
    curl \
    git \
    xz-utils \
    zip \
    unzip \
    libopencv-dev \
    libhdf5-dev \
    ca-certificates \
    ninja-build \
    build-essential \
    libjpeg-dev \
    libpng-dev \
    cmake \
    --no-install-recommends 

# Clean Up
RUN	apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.6.9 /uv /uvx /bin/

# To Add User and Group for vscode user
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID
ARG STORAGE_GID=1002
RUN	groupadd --gid $USER_GID $USERNAME \
    && useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && su $USERNAME \
    && cd $HOME
RUN groupadd -g $STORAGE_GID storage \
    && usermod -a -G storage $USERNAME
