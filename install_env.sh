#!/bin/bash

set -e

ARCH=$(uname -m)
echo "Detected system architecture: $ARCH"

# Select the corresponding miniforge installer based on architecture
# Network detection: if google.com is reachable = overseas, otherwise = China mainland
if curl -s --connect-timeout 2 https://www.google.com > /dev/null; then
    # Overseas: use official GitHub URL (faster)
    USE_CHINA_MIRROR=false
    echo "✅ Overseas network detected, using official download source"
    if [ "$ARCH" = "x86_64" ]; then
        MINIFORGE_URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
    elif [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
        MINIFORGE_URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh"
    fi
else
    # China mainland: automatically switch to Tsinghua mirror
    USE_CHINA_MIRROR=true
    echo "📶 China mainland network detected, automatically switching to Tsinghua mirror for acceleration"
    if [ "$ARCH" = "x86_64" ]; then
        MINIFORGE_URL="https://mirrors.tuna.tsinghua.edu.cn/github-release/conda-forge/miniforge/LatestRelease/Miniforge3-Linux-x86_64.sh"
    elif [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
        MINIFORGE_URL="https://mirrors.tuna.tsinghua.edu.cn/github-release/conda-forge/miniforge/LatestRelease/Miniforge3-Linux-aarch64.sh"
    fi
fi

# Check and clean up existing miniforge installation
if [ -d ~/miniforge3 ]; then
    echo "Existing Miniforge installation detected, removing..."
    rm -rf ~/miniforge3
fi

# Detect system type and install wget
echo "Detecting system type..."
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
    echo "Detected system: $OS"
else
    echo "Warning: Unable to detect system type, trying default method"
    OS="unknown"
fi

# Install dependencies based on system type
echo "Installing dependencies: wget net-tools unzip..."
if [[ "$OS" == "ubuntu" ]] || [[ "$OS" == "debian" ]]; then
    apt-get update
    apt-get install -y wget net-tools unzip
elif [[ "$OS" == "centos" ]] || [[ "$OS" == "rhel" ]] || [[ "$OS" == "rocky" ]] || [[ "$OS" == "almalinux" ]]; then
    yum install -y wget
else
    echo "Trying generic method to install wget..."
    if command -v apt-get &> /dev/null; then
        apt-get update && apt-get install -y wget
    elif command -v yum &> /dev/null; then
        yum install -y wget
    else
        echo "Error: Unable to determine package manager, please install wget manually"
        exit 1
    fi
fi

# Download and install miniforge
echo "Starting Miniforge download..."
wget "$MINIFORGE_URL" -O ~/miniforge3.sh

echo "Starting Miniforge installation..."
bash ~/miniforge3.sh -b -p ~/miniforge3

echo "Cleaning up installation files..."
rm -rf ~/miniforge3.sh

# Initialize conda
echo "Initializing conda..."
. ~/.bashrc
~/miniforge3/bin/conda init bash

# Configure pip mirror source
echo "Configuring pip mirror source..."
~/miniforge3/bin/pip config set global.index-url https://mirrors.tencent.com/pypi/simple/

# Create bench environment
echo "Creating bench environment..."
~/miniforge3/bin/conda create -y -n bench --clone root

echo "Activating bench environment and installing coscmd..."
# Use conda run to execute commands in the bench environment
# Install coscmd
~/miniforge3/bin/conda run -n bench pip install coscmd -i https://mirrors.tencent.com/pypi/simple/

# Install ansible
echo "Activating bench environment and installing ansible..."
~/miniforge3/bin/conda run -n bench pip install ansible

# ===================== pssh =====================
echo "Installing pssh (automatically selecting the fastest source)..."
if [ "$USE_CHINA_MIRROR" = true ]; then
    echo "📶 China mainland network: installing pssh via Gitee high-speed mirror"
    ~/miniforge3/bin/conda run -n bench pip install git+https://gitee.com/cxxsheng/pssh.git
else
    echo "✅ Overseas network: installing pssh via official GitHub"
    ~/miniforge3/bin/conda run -n bench pip install git+https://github.com/lilydjwg/pssh
fi
# ======================================================================

# ===================== Set auto-activate bench environment =====================
echo "Setting bench environment to auto-activate on login..."
echo "conda activate bench" >> ~/.bashrc
# ======================================================================
