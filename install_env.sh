#!/bin/bash

# 安装 miniforge3 pssh ansible coscmd

# 自动识别系统架构并安装对应的miniforge环境

set -e  # 遇到错误立即退出

# 检测系统架构
ARCH=$(uname -m)
echo "检测到系统架构: $ARCH"

# 根据架构选择对应的miniforge安装包
# 检测网络：能访问 google.com = 外网环境，否则 = 国内环境
if curl -s --connect-timeout 2 https://www.google.com > /dev/null; then
    # 外网：使用官方 GitHub 地址（速度快）
    USE_CHINA_MIRROR=false
    echo "✅ 检测到外网环境，使用官方下载源"
    if [ "$ARCH" = "x86_64" ]; then
        MINIFORGE_URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
    elif [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
        MINIFORGE_URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh"
    fi
else
    # 国内：自动切清华镜像
    USE_CHINA_MIRROR=true
    echo "📶 检测到国内网络，自动切换清华镜像加速"
    if [ "$ARCH" = "x86_64" ]; then
        MINIFORGE_URL="https://mirrors.tuna.tsinghua.edu.cn/github-release/conda-forge/miniforge/LatestRelease/Miniforge3-Linux-x86_64.sh"
    elif [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
        MINIFORGE_URL="https://mirrors.tuna.tsinghua.edu.cn/github-release/conda-forge/miniforge/LatestRelease/Miniforge3-Linux-aarch64.sh"
    fi
fi

# 检查并清理已存在的miniforge安装
if [ -d ~/miniforge3 ]; then
    echo "检测到已存在的 Miniforge 安装，正在删除..."
    rm -rf ~/miniforge3
fi

# 检测系统类型并安装 wget
echo "检测系统类型..."
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
    echo "检测到系统: $OS"
else
    echo "警告: 无法检测系统类型，尝试使用默认方式"
    OS="unknown"
fi

# 根据系统类型安装依赖
echo "安装依赖 wget net-tools unzip..."
if [[ "$OS" == "ubuntu" ]] || [[ "$OS" == "debian" ]]; then
    apt-get update
    apt-get install -y wget net-tools unzip
elif [[ "$OS" == "centos" ]] || [[ "$OS" == "rhel" ]] || [[ "$OS" == "rocky" ]] || [[ "$OS" == "almalinux" ]]; then
    yum install -y wget
else
    echo "尝试使用通用方式安装 wget..."
    if command -v apt-get &> /dev/null; then
        apt-get update && apt-get install -y wget
    elif command -v yum &> /dev/null; then
        yum install -y wget
    else
        echo "错误: 无法确定包管理器，请手动安装 wget"
        exit 1
    fi
fi

# 下载并安装miniforge
echo "开始下载 Miniforge..."
wget "$MINIFORGE_URL" -O ~/miniforge3.sh

echo "开始安装 Miniforge..."
bash ~/miniforge3.sh -b -p ~/miniforge3

echo "清理安装文件..."
rm -rf ~/miniforge3.sh

# 初始化conda
echo "初始化 conda..."
. ~/.bashrc
~/miniforge3/bin/conda init bash

# 配置pip镜像源
echo "配置 pip 镜像源..."
~/miniforge3/bin/pip config set global.index-url https://mirrors.tencent.com/pypi/simple/

# 创建bench环境
echo "创建 bench 环境..."
~/miniforge3/bin/conda create -y -n bench --clone root

echo "激活 bench 环境并安装 cos..."
# 使用 conda run 在 bench 环境中执行命令
# 安装 coscmd
~/miniforge3/bin/conda run -n bench pip install coscmd -i https://mirrors.tencent.com/pypi/simple/

# 安装 ansible
echo "激活 bench 环境并安装 ansible..."
~/miniforge3/bin/conda run -n bench pip install ansible

# ===================== pssh =====================
echo "正在安装 pssh（自动选择最快源）..."
if [ "$USE_CHINA_MIRROR" = true ]; then
    echo "📶 国内网络：使用 Gitee 高速镜像安装 pssh"
    ~/miniforge3/bin/conda run -n bench pip install git+https://gitee.com/cxxsheng/pssh.git
else
    echo "✅ 外网环境：使用官方 GitHub 安装 pssh"
    ~/miniforge3/bin/conda run -n bench pip install git+https://github.com/lilydjwg/pssh
fi
# ======================================================================

# ===================== 设置自动激活 bench 环境 =====================
echo "设置每次登录自动激活 bench 环境..."
echo "conda activate bench" >> ~/.bashrc
# ======================================================================
