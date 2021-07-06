import torch
from torch.utils.data import DataLoader
import numpy as np

class MaskTextDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, label_id_dic, mask, is_test):
        """
        用于返回数据集中数据
        file_path: 存储数据集信息的文件路径，每个视频对应视频特征、音频特征、图片特征、文本特征的存储路径、以及对应的标签 
        label_id_dic: lable 转 id字典文件
        is_train: 区分训练和测试
        """
        self.is_test = is_test
        self.mask = mask == 1
        with open(file_path, 'r') as f:
            data = f.read().splitlines()
        if(not is_test):
            self.video_path = data[0::6]
            self.audio_path = data[1::6]
            self.image_path = data[2::6]
            self.text_path = data[3::6]
            self.labels = data[4::6]
        else:
            self.video_path = data[0::5]
            self.audio_path = data[1::5]
            self.image_path = data[2::5]
            self.text_path = data[3::5]

        self.label_id_dic = label_id_dic

    def __getitem__(self, idx):
        data = {}
        # ---------text to token------------
        text_file = self.text_path[idx]
        dic = eval(open(text_file.replace(
            '..', '/home/tione/notebook/multimodal/')).read())
        asr = ((dic['video_asr']).replace('|', ','))
        ocr = ((dic['video_ocr']).replace('|', ','))
        data['asr'] = asr
        data['ocr'] = ocr
        data['video_path'] = self.video_path[idx].split(
                "/")[-1].split(".")[0] + '.mp4'  # 返回.mp4文件名
        # ----------label to id--------------
        if(not self.is_test):
            ids = torch.zeros(82)
            labels = self.labels[idx]
            labels = labels.split(',')
            for label in labels:
                ids[self.label_id_dic[label]] = 1
            data['labels'] = ids[self.mask]
        return data

    def __len__(self):
        return len(self.video_path)


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, label_id_dic, is_test):
        """
        用于返回数据集中数据
        file_path: 存储数据集信息的文件路径，每个视频对应视频特征、音频特征、图片特征、文本特征的存储路径、以及对应的标签 
        label_id_dic: lable 转 id字典文件
        is_train: 区分训练和测试
        """
        self.is_test = is_test
        with open(file_path, 'r') as f:
            data = f.read().splitlines()
        if(not is_test):
            self.video_path = data[0::6]
            self.audio_path = data[1::6]
            self.image_path = data[2::6]
            self.text_path = data[3::6]
            self.labels = data[4::6]
        else:
            self.video_path = data[0::5]
            self.audio_path = data[1::5]
            self.image_path = data[2::5]
            self.text_path = data[3::5]

        self.label_id_dic = label_id_dic

    def __getitem__(self, idx):
        data = {}
        # ---------text to token------------
        text_file = self.text_path[idx]
        dic = eval(open(text_file.replace(
            '..', '/home/tione/notebook/multimodal/')).read())
        asr = ((dic['video_asr']).replace('|', ','))
        ocr = ((dic['video_ocr']).replace('|', ','))
        data['asr'] = asr
        data['ocr'] = ocr
        data['video_path'] = self.video_path[idx].split(
                "/")[-1].split(".")[0] + '.mp4'  # 返回.mp4文件名
        # ----------label to id--------------
        if(not self.is_test):
            ids = torch.zeros(82)
            labels = self.labels[idx]
            labels = labels.split(',')
            for label in labels:
                ids[self.label_id_dic[label]] = 1
            data['labels'] = ids
        return data

    def __len__(self):
        return len(self.video_path)
    
class TextVideoDataset(torch.utils.data.Dataset):
    def __init__(self, lable_path, text_path, video_path, video_path_path, is_test = False):
        """
        用于返回数据集中数据
        file_path: 存储数据集信息的文件路径，每个视频对应视频特征、音频特征、图片特征、文本特征的存储路径、以及对应的标签 
        label_id_dic: lable 转 id字典文件
        is_train: 区分训练和测试
        """
        self.is_test = is_test
        if(not is_test):
            self.labels = np.load(lable_path)
        self.text_features =  np.load(text_path)
        self.video_features =  np.load(video_path)
        self.video_path = np.load(video_path_path)

    def __getitem__(self, idx):
        data = {}
        data['video_features'] = self.video_features[idx]
        data['text_features'] = self.text_features[idx]
        data['video_path'] = self.video_path[idx]
        if(not self.is_test):
            data['labels'] = self.labels[idx]
        return data

    def __len__(self):
        return self.video_features.shape[0]

class TextVideoAsrDataset(torch.utils.data.Dataset):
    def __init__(self, lable_path, text_path, video_path, asr_path, video_path_path, is_test = False):
        """
        用于返回数据集中数据
        file_path: 存储数据集信息的文件路径，每个视频对应视频特征、音频特征、图片特征、文本特征的存储路径、以及对应的标签 
        label_id_dic: lable 转 id字典文件
        is_train: 区分训练和测试
        """
        self.is_test = is_test
        if(not is_test):
            self.labels = np.load(lable_path)
        self.text_features =  np.load(text_path)
        self.video_features =  np.load(video_path)
        self.asr_features =  np.load(asr_path)
        self.video_path = np.load(video_path_path)

    def __getitem__(self, idx):
        data = {}
        data['video_features'] = self.video_features[idx]
        data['text_features'] = self.text_features[idx]
        data['asr_features'] = self.asr_features[idx]
        data['video_path'] = self.video_path[idx]
        if(not self.is_test):
            data['labels'] = self.labels[idx]
        return data

    def __len__(self):
        return self.video_features.shape[0]

def my_text_video_dataloader(lable_path, text_path, video_path, video_path_path, is_test, batch_size, shuffle):
    dataset = TextVideoDataset(lable_path, text_path, video_path, video_path_path, is_test)
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)

def my_text_video_asr_dataloader(lable_path, text_path, video_path, asr_path, video_path_path, is_test, batch_size, shuffle):
    dataset = TextVideoAsrDataset(lable_path, text_path, video_path, asr_path, video_path_path, is_test)
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)

def my_dataloader(file_path, label_id_dic, is_test, batch_size, shuffle):
    dataset = TextDataset(file_path, label_id_dic, is_test)
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)


def my_mask_dataloader(file_path, label_id_dic, mask, is_test, batch_size, shuffle):
    dataset = MaskTextDataset(file_path, label_id_dic, mask, is_test)
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)