# install environment
CONDA_NEW_ENV=tagging
JUPYTER_ROOT=/home/tione/notebook

# ubuntu system library
sudo apt-get update
sudo apt-get install -y apt-utils
sudo apt-get install -y libsndfile1-dev ffmpeg
  
conda create --prefix ${JUPYTER_ROOT}/envs/${CONDA_NEW_ENV} -y cudatoolkit=10.0 cudnn=7.6.0 python=3.9 ipykernel
conda activate ${JUPYTER_ROOT}/envs/${CONDA_NEW_ENV}


# pytorch
pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

# x3d environment
pip install numpy
pip install 'git+https://github.com/facebookresearch/fvcore'
pip install simplejson
conda install av -c conda-forge
pip install -U iopath
pip install psutil
pip install opencv-python
pip install tensorboard
conda install -c conda-forge moviepy
pip install pytorchvideo
pip install -U cython
python -m pip install -e ./utils/detectron2
# pyslowfast
export PYTHONPATH=/home/tione/notebook/algo-2021-jbtjjsw/slowFast/slowfast:$PYTHONPATH
cd ./utils/slowFast
python setup.py build develop
cd -
pip install timm