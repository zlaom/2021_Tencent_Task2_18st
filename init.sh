#!/usr/bin/env bash
# #################### get env directories
# CONDA_ROOT
CONDA_CONFIG_ROOT_PREFIX=$(conda config --show root_prefix)
echo "CONDA_CONFIG_ROOT_PREFIX= ${CONDA_CONFIG_ROOT_PREFIX}"
get_conda_root_prefix() {
  TMP_POS=$(awk -v a="${CONDA_CONFIG_ROOT_PREFIX}" -v b="/" 'BEGIN{print index(a, b)}')
  TMP_POS=$((TMP_POS-1))
  if [ $TMP_POS -ge 0 ]; then
    echo "${CONDA_CONFIG_ROOT_PREFIX:${TMP_POS}}"
  else
    echo ""
  fi
}
CONDA_ROOT=$(get_conda_root_prefix)
if [ ! -d "${CONDA_ROOT}" ]; then
  echo "CONDA_ROOT= ${CONDA_ROOT}, not exists, exit"
  exit 1
fi
# CONDA ENV
CONDA_NEW_ENV=taac2021-tagging-jbtjjsw
# JUPYTER_ROOT
JUPYTER_ROOT=/home/tione/notebook
if [ ! -d "${JUPYTER_ROOT}" ]; then
  echo "JUPYTER_ROOT= ${JUPYTER_ROOT}, not exists, exit"
  exit 1
fi
# CODE ROOT
CODE_ROOT=${JUPYTER_ROOT}/algo-2021-jbtjjsw
if [ ! -d "${CODE_ROOT}" ]; then
  echo "CODE_ROOT= ${CODE_ROOT}, not exists, exit"
  exit 1
fi
# OS RELEASE
OS_ID=$(awk -F= '$1=="ID" { print $2 ;}' /etc/os-release)
OS_ID=${OS_ID//"\""/""}

echo "CONDA_ROOT= ${CONDA_ROOT}"
echo "CONDA_NEW_ENV= ${CONDA_NEW_ENV}"
echo "JUPYTER_ROOT= ${JUPYTER_ROOT}"
echo "CODE_ROOT= ${CODE_ROOT}"
echo "OS_ID= ${OS_ID}"

# #################### obviously set $1 to be 'run' to run ./init.sh
if [ -z "$1" ]; then
  ACTION="check"
else
  ACTION=$(echo "$1" | tr '[:upper:]' '[:lower:]')
fi
if [ "${ACTION}" != "run" ]; then
  echo "[Info] you don't set the ACTION as 'run', so just check the environment"
  exit 0
fi

# #################### install system libraries
if [ "${OS_ID}" == "ubuntu" ]; then
  echo "[Info] installing system libraries in ${OS_ID}"
  sudo apt-get update
  sudo apt-get install -y apt-utils
  sudo apt-get install -y libsndfile1-dev ffmpeg
elif [ "${OS_ID}" == "centos" ]; then
  echo "[Info] installing system libraries in ${OS_ID}"
  yum install -y libsndfile libsndfile-devel ffmpeg ffmpeg-devel
else
  echo "[Warning] os not supported for ${OS_ID}"
  exit 1
fi

# #################### use tsinghua conda sources
conda config --show channels
# conda config --remove channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
# conda config --remove channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
# conda config --remove channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
# conda config --show channels
# conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
# conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
# conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
# conda config --set show_channel_urls yes
# conda config --show channels

# #################### create conda env and activate
# conda in shell propagation issue - https://stackoverflow.com/questions/52779016/conda-command-working-in-command-prompt-but-not-in-bash-script/52813960#52813960
# shellcheck source=/opt/conda/etc/profile.d/conda.sh
source "${CONDA_ROOT}/etc/profile.d/conda.sh"

# ###### create env and activate
# Pytorch 1.7.1 GPU dependencies - https://pytorch.org/get-started/locally/

# create env by prefix
echo "[Conda install]"
conda create -p ${JUPYTER_ROOT}/envs/${CONDA_NEW_ENV} -y python=3.8 ipykernel
conda activate ${JUPYTER_ROOT}/envs/${CONDA_NEW_ENV}
# create env by name
# conda create -n ${CONDA_NEW_ENV} -y cudatoolkit=10.0 cudnn=7.6.0 python=3.7 ipykernel
# conda activate ${CONDA_NEW_ENV}

conda info --envs

# #################### create jupyter kernel
# create a kernel for conda env
python -m ipykernel install --user --name ${CONDA_NEW_ENV} --display-name "TAAC2021 (${CONDA_NEW_ENV})"

# #################### install python libraries
pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install transformers opencv-python

# check tensorflow GPU
# python -c "import torch; print(torch.__version__)"

# check library versions
echo "[Pytorch]"
python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.cuda.is_available())"
echo "[NumPy]"
python -c "import numpy as np; print(np.__version__)"
echo "[OpenCV]"
python -c "import cv2; print(cv2.__version__)"
echo "[Transformers]"
python -c "import transformers; print(transformers.__version__)"