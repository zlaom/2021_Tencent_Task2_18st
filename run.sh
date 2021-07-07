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
# DATASET ROOT
echo ["Link dataset"]
DATASET_ROOT=/home/tione/notebook/algo-2021/dataset/
ln -s ${DATASET_ROOT} ${CODE_ROOT}


DATASET_ROOT=${CODE_ROOT}/dataset
if [ ! -d "${DATASET_ROOT}" ]; then
  echo "DATASET_ROOT= ${DATASET_ROOT}, not exists, exit"
  exit 1
fi
# OS RELEASE
OS_ID=$(awk -F= '$1=="ID" { print $2 ;}' /etc/os-release)

echo "CONDA_ROOT= ${CONDA_ROOT}"
echo "CONDA_NEW_ENV= ${CONDA_NEW_ENV}"
echo "JUPYTER_ROOT= ${JUPYTER_ROOT}"
echo "CODE_ROOT= ${CODE_ROOT}"
echo "DATASET_ROOT= ${DATASET_ROOT}"
echo "OS_ID= ${OS_ID}"

# #################### activate conda env and check lib versions
# solve run problem in Jupyter Notebook
# conda in shell propagation issue - https://stackoverflow.com/questions/52779016/conda-command-working-in-command-prompt-but-not-in-bash-script/52813960#52813960
CONDA_CONFIG_FILE="${CONDA_ROOT}/etc/profile.d/conda.sh"
if [ ! -f "${CONDA_CONFIG_FILE}" ]; then
  echo "CONDA_CONFIG_FILE= ${CONDA_CONFIG_FILE}, not exists, exit"
  exit 1
fi
# shellcheck disable=SC1090
source "${CONDA_CONFIG_FILE}"

# ###### activate conda env
# conda env by name
# conda activate ${CONDA_NEW_ENV}
# conda env by prefix
conda activate ${JUPYTER_ROOT}/envs/${CONDA_NEW_ENV}
conda info --envs

# check library versions
echo "[Pytorch version]"
python -c "import torch; print(torch.__version__)"
echo "[Cuda available]"
python -c "import torch; print(torch.cuda.is_available())"
echo "[NumPy]"
python -c "import numpy as np; print(np.__version__)"
echo "[OpenCV]"
python -c "import cv2; print(cv2.__version__)"
echo "[Transformers]"
python -c "import transformers; print(transformers.__version__)"

# #################### get 1st input argument as TYPE
TYPE=train
if [ -z "$1" ]; then
    echo "[Warning] TYPE is not set, using 'train' as default"
else
    TYPE=$(echo "$1" | tr '[:upper:]' '[:lower:]')
    echo "[Info] TYPE is ${TYPE}"
fi

# #################### execute according to TYPE
########## train
if [ "$TYPE" = "train" ]; then
#     cd src/text_predict/ocr_bert_base
#     time python ocr_classifier.py
#     cd -
    cd src/text_predict/ocr_bert_class_train
#     time python train.py
    cd -
    
    cd src/weight_fusion
    time python train.py
    cd -
  exit 0

########## test
elif [ "$TYPE" = "test" ]; then
    cd src/text_predict/ocr_bert_class_train
    time python predict.py
    cd -
    cd src/weight_fusion
    time python predict.py
    cd -
  exit 0
# ######### text predict
fi