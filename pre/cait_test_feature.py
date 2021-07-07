import os
import time
import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as T
# 注意安装 pip install timm
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

# test
mode = 'test'
# 序号
series = 0

# 视频存放路径
absolute_dir = '/home/tione/notebook/algo-2021/dataset/videos/test_5k_2nd'
# 特征保存路径
save_dir = '/home/tione/notebook/dataset/features/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)  
# 加载模型
device_id = 'cuda:0'

# baseline的datafile存放位置
test_datafile = './utils/datafile/baseline/test.txt'

# label_id.txt文件位置
label_id_file = './utils/label_id.txt'

# 模型的位置
ppath = './pre/checkpoint/cait_m48.pth'
if not os.path.exists(ppath):
    model = timm.create_model('cait_m48_448', pretrained=True)
    torch.save(model, './pre/checkpoint/cait_m48.pth')
ckpt = './pre/checkpoint/cait_m48.pth'


def read_lines(datafile):
    with open(datafile, 'r') as f:
        pathfile = f.read().splitlines()
    return pathfile

def get_basename(path):
    # return the file basename
    return os.path.basename(path).split('.')[0]

def save_to_file(filename, datafile):
    # save datafile to local directory
    with open(filename, 'w') as f:
        for line in datafile:
            f.write(line+'\n')
            

# 创建labels--ids转换字典
def get_label2id_dic(path):
    label2id_dic = {}
    with open(path, 'r') as f:
        while True: 
            line = f.readline()
            if line == '':
                break
            pair = line.strip().split('\t')
            label2id_dic[pair[0]] = int(pair[1])
    return label2id_dic

label2id_dic = get_label2id_dic(label_id_file)

def label2id(label2id_dic, labels):
    ids = [0 for i in range(82)]
    labels = labels.split(',')
    for label in labels:
        ids[label2id_dic[label]] = 1
    return ids

def get_test_transforms(input_size):
    mean, std = [0.485, 0.456, 0.406],[0.229, 0.224, 0.225]    
    transformations = {}
    transformations= T.Compose(
        [T.Resize(input_size, interpolation=3),
         T.CenterCrop(input_size),
         T.ToTensor(),
         T.Normalize(mean, std)])
    return transformations

# create the data transform that cait expects
transform = get_test_transforms(448)

pathfile = read_lines(test_datafile)
test_path = pathfile[0::5]


device = torch.device(device_id) if torch.cuda.is_available() else torch.device('cpu'); print(device)
model = torch.load(ckpt)
model.to(device)
model.eval()

paths = test_path

# 创建对应的文件夹
save_path = os.path.join(save_dir, mode)
if not os.path.exists(save_path):
    os.makedirs(save_path)

datafile = []
for idx, path in enumerate(paths):
    tic = time.time()
    
    basename = get_basename(path) # basename，为了得到视频的名字
    video = os.path.join(absolute_dir, basename+'.mp4') # 由basename推出视频存放路径
    
    # 打开视频
    cap = cv2.VideoCapture()
    if not cap.open(video):
        print('cannot open video:{}'.format(video))
    # 计算采样的位置
    n_frames = cap.get(7)
    fps = cap.get(5)
    times = n_frames / fps
    # 每秒提几帧
    N = 4
    step = fps // N
    
    # 获取要提帧的索引
    index = []
    for sec in range(int(times) + 1):
        for i in range(N):
            new_index = sec * fps + i * step
            if new_index > n_frames:
                break
            index.append(new_index)
    
    # 开始提特征
    features = []
    i_image = 0
    i_frame = 0
    while True:
        has_frame, frame = cap.read()
        if has_frame == False:
            break
        if i_frame in index:
            ##------------提特征------------
            img = Image.fromarray(frame, mode='RGB') # 把数组转成PIL格式
            img = transform(img).unsqueeze(0).to(device)
            feature = model.forward_features(img)
            features.append(feature[0].detach().cpu().numpy())
            del feature
            ##-----------一帧结束------------
            i_image += 1
        i_frame += 1
    cap.release()
    
    # 保存npy文件
    features = np.array(features)
    save_path = os.path.join(save_dir, mode, basename+'.npy')
    np.save(save_path, features)
    
    # 保存datafile的内容
    datafile.append(save_path)
    datafile.append('')
    datafile.append('')
    
    toc = time.time()
    print(idx, toc-tic)
    
# 保存datafile
datafile_save_dir = './utils/datafile/features'
if not os.path.exists(datafile_save_dir):
    os.makedirs(datafile_save_dir)   
save_to_file(os.path.join(datafile_save_dir, mode+'.txt'), datafile)