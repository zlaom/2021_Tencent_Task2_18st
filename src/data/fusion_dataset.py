import torch
import numpy as np 

# text data; video features
class TdVf_Dataset(torch.utils.data.Dataset):
    def __init__(self, datafile, label_id_dic, video_path, is_train):
        with open(datafile, 'r') as f:
            path_file = f.read().splitlines()

        self.text_path = path_file[3::6]
        self.labels = path_file[4::6]
        
        self.video_features = np.load(video_path)
        
        self.label_id_dic = label_id_dic
        self.is_train = is_train

    def __getitem__(self, idx):
        data = {}
        ## ---------text to token------------ 
        text_file = self.text_path[idx]
        dic = eval(open(text_file).read())
        sentences = (dic['video_ocr'].replace('|', '，').replace(' ', ''))
        data['text'] = sentences
        ##-----------video feature-----------
        data['video'] = torch.tensor(self.video_features[idx])
        ## ----------label to id-------------- 
        ids = torch.zeros(82)
        labels = self.labels[idx]
        labels = labels.split(',')
        for label in labels:
            ids[self.label_id_dic[label]] = 1
        
        data['labels'] = ids
        if self.is_train == False:
            data['video_path'] = self.text_path[idx].split("/")[-1].split(".")[0] + '.mp4' # 返回.mp4文件名
            
        return data

    def __len__(self):
        return len(self.text_path)
    
    
# text features, video features
class NVDataset(torch.utils.data.Dataset):
    def __init__(self, datafile, label_id_dic, text_path, video_path, is_train):
        # baseline datafile for get labels and video names
        with open(datafile, 'r') as f:
            path_file = f.read().splitlines()
        if is_train:
            del path_file[2622 * 6 : 2623 * 6]
        self.text_path = path_file[3::6]    
        self.labels = path_file[4::6]

        self.text_features = np.load(text_path)
        self.video_features = np.load(video_path)
        if is_train:
            self.video_features = np.delete(self.video_features, 2622, axis=0)
            
        self.label_id_dic = label_id_dic
        self.is_train = is_train

    def __getitem__(self, idx):
        data = {}
        ## ---------text to token------------ 
        data['text'] = torch.tensor(self.text_features[idx])
        ##-----------video feature-----------
        data['video'] = torch.tensor(self.video_features[idx])
        ## ----------label to id-------------- 
        ids = torch.zeros(82)
        labels = self.labels[idx]
        labels = labels.split(',')
        for label in labels:
            ids[self.label_id_dic[label]] = 1
        data['labels'] = ids
        ## ----------video name---------------
        if self.is_train == False:
            data['video_path'] = self.text_path[idx].split("/")[-1].split(".")[0] + '.mp4' # 返回.mp4文件名
            
        return data

    def __len__(self):
        return len(self.text_path)
    

class TVDataset(torch.utils.data.Dataset):
    def __init__(self, datafile, label_id_dic, text_path, video_path, is_train):
        # baseline datafile for get labels and video names
        with open(datafile, 'r') as f:
            path_file = f.read().splitlines()

        self.text_path = path_file[3::6]    
        self.labels = path_file[4::6]
            
        self.text_features = np.load(text_path)
        self.video_features = np.load(video_path)
            
        self.label_id_dic = label_id_dic
        self.is_train = is_train

    def __getitem__(self, idx):
        data = {}
        ## ---------text to token------------ 
        data['text'] = torch.tensor(self.text_features[idx])
        ##-----------video feature-----------
        data['video'] = torch.tensor(self.video_features[idx])
        ## ----------label to id-------------- 
        ids = torch.zeros(82)
        labels = self.labels[idx]
        labels = labels.split(',')
        for label in labels:
            ids[self.label_id_dic[label]] = 1
        data['labels'] = ids
        ## ----------video name---------------
        if self.is_train == False:
            data['video_path'] = self.text_path[idx].split("/")[-1].split(".")[0] + '.mp4' # 返回.mp4文件名
            
        return data

    def __len__(self):
        return len(self.text_path)
    

class TNDataset(torch.utils.data.Dataset):
    def __init__(self, datafile, label_id_dic, text_path, video_path, is_train):
        # baseline datafile for get labels and video names
        with open(datafile, 'r') as f:
            path_file = f.read().splitlines()
        if is_train:
            del path_file[2622 * 6 : 2623 * 6]   
        self.text_path = path_file[3::6]    
        self.labels = path_file[4::6]

            
        self.text_features = np.load(text_path)
        self.video_features = np.load(video_path)
        if is_train:
            self.text_features = np.delete(self.text_features, 2622, axis=0)
            
        self.label_id_dic = label_id_dic
        self.is_train = is_train

    def __getitem__(self, idx):
        data = {}
        ## ---------text to token------------ 
        data['text'] = torch.tensor(self.text_features[idx])
        ##-----------video feature-----------
        data['video'] = torch.tensor(self.video_features[idx])
        ## ----------label to id-------------- 
        ids = torch.zeros(82)
        labels = self.labels[idx]
        labels = labels.split(',')
        for label in labels:
            ids[self.label_id_dic[label]] = 1
        data['labels'] = ids
        ## ----------video name---------------
        if self.is_train == False:
            data['video_path'] = self.text_path[idx].split("/")[-1].split(".")[0] + '.mp4' # 返回.mp4文件名
            
        return data

    def __len__(self):
        return len(self.text_path)
    
class TNVDataset(torch.utils.data.Dataset):
    def __init__(self, datafile, label_id_dic, text_path, video_path, nextvlad_path, is_train):
        # baseline datafile for get labels and video names
        with open(datafile, 'r') as f:
            path_file = f.read().splitlines()
        if is_train:
            del path_file[2622 * 6 : 2623 * 6]   
        self.text_path = path_file[3::6]    
        self.labels = path_file[4::6]

        self.text_features = np.load(text_path)
        self.video_features = np.load(video_path)
        self.nextvlad_features = np.load(nextvlad_path)
        if is_train:
            self.text_features = np.delete(self.text_features, 2622, axis=0)
            self.video_features = np.delete(self.video_features, 2622, axis=0)
            
        self.label_id_dic = label_id_dic
        self.is_train = is_train

    def __getitem__(self, idx):
        data = {}
        ## ---------text to token------------ 
        data['text'] = torch.tensor(self.text_features[idx])
        ##-----------video feature-----------
        data['video'] = torch.tensor(self.video_features[idx])
        ##-----------nextvlad feature-----------
        data['nextvlad'] = torch.tensor(self.nextvlad_features[idx])
        ## ----------label to id-------------- 
        ids = torch.zeros(82)
        labels = self.labels[idx]
        labels = labels.split(',')
        for label in labels:
            ids[self.label_id_dic[label]] = 1
        data['labels'] = ids
        ## ----------video name---------------
        if self.is_train == False:
            data['video_path'] = self.text_path[idx].split("/")[-1].split(".")[0] + '.mp4' # 返回.mp4文件名
            
        return data

    def __len__(self):
        return len(self.text_path)
    
class TNV2Dataset(torch.utils.data.Dataset):
    def __init__(self, datafile, label_id_dic, text_path, video_path, nextvlad_path, is_train):
        # baseline datafile for get labels and video names
        with open(datafile, 'r') as f:
            path_file = f.read().splitlines()
        if is_train:
            del path_file[2622 * 6 : 2623 * 6]   
        self.text_path = path_file[3::6]    
        self.labels = path_file[4::6]

        self.text_features = np.load(text_path)
        self.video_features = np.load(video_path)
        self.nextvlad_features = np.load(nextvlad_path)
        if is_train:
            self.text_features = np.delete(self.text_features, 2622, axis=0)
            self.video_features = np.delete(self.video_features, 2622, axis=0)
            
        self.label_id_dic = label_id_dic
        self.is_train = is_train

    def __getitem__(self, idx):
        data = {}
        ## ---------text to token------------ 
        data['text'] = torch.tensor(self.text_features[idx])
        ##-----------video feature-----------
        data['video'] = torch.tensor(self.video_features[idx])
        ##-----------nextvlad feature-----------
        data['nextvlad'] = torch.tensor(self.nextvlad_features[idx])
        ## ----------label to id-------------- 
        ids = torch.zeros(82)
        labels = self.labels[idx]
        labels = labels.split(',')
        for label in labels:
            ids[self.label_id_dic[label]] = 1
        data['labels'] = ids
        ## ----------video name---------------
        if self.is_train == False:
            data['video_path'] = self.text_path[idx].split("/")[-1].split(".")[0] + '.mp4' # 返回.mp4文件名
            
        return data

    def __len__(self):
        return len(self.text_path)


# # style features
# class Style_Dataset(torch.utils.data.Dataset):
#     def __init__(self, datafile, label_id_dic, style_paths, is_train):
#         # style_paths为一个字典
#         # baseline datafile for get labels and video names
#         with open(datafile, 'r') as f:
#             path_file = f.read().splitlines()
        
        
#         self.text_path = path_file[3::6]    
#         self.labels = path_file[4::6]
        
#         self.video_features = np.load(style_paths['video'])
        
#         self.scene_features = np.load(style_paths['scene'])
#         self.people_features = np.load(style_paths['people'])
#         self.display_features = np.load(style_paths['display'])
#         self.global_features = np.load(style_paths['global'])
#         self.text_features = np.load(style_paths['text'])
        
#         self.label_id_dic = label_id_dic
#         self.is_train = is_train

#     def __getitem__(self, idx):
#         data = {}
#         ##-----------text feature------------
#         data['text'] = torch.tensor(self.text_features[idx])
#         ##-----------video feature-----------
#         data['video'] = torch.tensor(self.video_features[idx])
#         ##-----------style features----------
#         data['scene'] = torch.tensor(self.scene_features[idx])
#         data['people'] = torch.tensor(self.people_features[idx])
#         data['display'] = torch.tensor(self.display_features[idx])
#         data['global'] = torch.tensor(self.global_features[idx])
#         ## ----------label to id-------------- 
#         ids = torch.zeros(82)
#         labels = self.labels[idx]
#         labels = labels.split(',')
#         for label in labels:
#             ids[self.label_id_dic[label]] = 1
#         data['labels'] = ids
#         ## ----------video name---------------
#         if self.is_train == False:
#             data['video_path'] = self.text_path[idx].split("/")[-1].split(".")[0] + '.mp4' # 返回.mp4文件名
            
#         return data

#     def __len__(self):
#         return len(self.text_path)


# style features
class Style_Dataset2(torch.utils.data.Dataset):
    def __init__(self, datafile, label_id_dic, style_paths, is_train):
        # style_paths为一个字典
        # baseline datafile for get labels and video names
        with open(datafile, 'r') as f:
            path_file = f.read().splitlines()
        
        
        self.text_path = path_file[3::6]    
        self.labels = path_file[4::6]
        
        self.video_features = np.load(style_paths['video'])
        self.text_features = np.load(style_paths['text'])
        
        self.scene_features = np.load(style_paths['scene'])
        self.people_features = np.load(style_paths['people'])
        self.display_features = np.load(style_paths['display'])
        self.global_features = np.load(style_paths['global'])
        
        
        self.label_id_dic = label_id_dic
        self.is_train = is_train

    def __getitem__(self, idx):
        data = {}
        ##-----------text feature------------
        data['text'] = torch.tensor(self.text_features[idx])
        ##-----------video feature-----------
        data['video'] = torch.tensor(self.video_features[idx])
        ##-----------style features----------
        data['scene'] = torch.tensor(self.scene_features[idx])
        data['people'] = torch.tensor(self.people_features[idx])
        data['display'] = torch.tensor(self.display_features[idx])
        data['global'] = torch.tensor(self.global_features[idx])
        ## ----------label to id-------------- 
        ids = torch.zeros(82)
        labels = self.labels[idx]
        labels = labels.split(',')
        for label in labels:
            ids[self.label_id_dic[label]] = 1
        data['labels'] = ids
        ## ----------video name---------------
        if self.is_train == False:
            data['video_path'] = self.text_path[idx].split("/")[-1].split(".")[0] + '.mp4' # 返回.mp4文件名
            
        return data

    def __len__(self):
        return len(self.text_path)

    
    

class Test_Style_Dataset2(torch.utils.data.Dataset):
    def __init__(self, datafile, label_id_dic, style_paths, is_train):
        # style_paths为一个字典
        # baseline datafile for get labels and video names
        with open(datafile, 'r') as f:
            path_file = f.read().splitlines()
        
        
        self.text_path = path_file[3::5]    
        
        self.video_features = np.load(style_paths['video'])
        self.text_features = np.load(style_paths['text'])
        
        self.scene_features = np.load(style_paths['scene'])
        self.people_features = np.load(style_paths['people'])
        self.display_features = np.load(style_paths['display'])
        self.global_features = np.load(style_paths['global'])
        
        
        self.label_id_dic = label_id_dic
        self.is_train = is_train

    def __getitem__(self, idx):
        data = {}
        ##-----------text feature------------
        data['text'] = torch.tensor(self.text_features[idx])
        ##-----------video feature-----------
        data['video'] = torch.tensor(self.video_features[idx])
        ##-----------style features----------
        data['scene'] = torch.tensor(self.scene_features[idx])
        data['people'] = torch.tensor(self.people_features[idx])
        data['display'] = torch.tensor(self.display_features[idx])
        data['global'] = torch.tensor(self.global_features[idx])
        ## ----------video name---------------
        if self.is_train == False:
            data['video_path'] = self.text_path[idx].split("/")[-1].split(".")[0] + '.mp4' # 返回.mp4文件名
            
        return data

    def __len__(self):
        return len(self.text_path)

    
    
class TestDataset(torch.utils.data.Dataset):
    def __init__(self, datafile, label_id_dic, text_path, video_path, is_train):
        # baseline datafile for get labels and video names
        with open(datafile, 'r') as f:
            path_file = f.read().splitlines()
        self.text_path = path_file[3::5]    

        self.text_features = np.load(text_path)
        self.video_features = np.load(video_path)
        
        self.label_id_dic = label_id_dic
        self.is_train = is_train

    def __getitem__(self, idx):
        data = {}
        ## ---------text to token------------ 
        data['text'] = torch.tensor(self.text_features[idx])
        ##-----------video feature-----------
        data['video'] = torch.tensor(self.video_features[idx])
        ## ----------video name---------------
        if self.is_train == False:
            data['video_path'] = self.text_path[idx].split("/")[-1].split(".")[0] + '.mp4' # 返回.mp4文件名
            
        return data

    def __len__(self):
        return len(self.text_path)
    
    
# text features, video features
class TfVd_Dataset(torch.utils.data.Dataset):
    def __init__(self, video_datafile, label_id_dic, text_path, is_train):
        # baseline datafile for get labels and video names
        with open(video_datafile, 'r') as f:
            path_file = f.read().splitlines()
        self.video_path = path_file[0::3]
        self.labels = path_file[1::3]

        self.text_features = np.load(text_path)
        
        self.label_id_dic = label_id_dic
        self.is_train = is_train

    def __getitem__(self, idx):
        data = {}
        ## ---------text to token------------ 
        data['text'] = torch.tensor(self.text_features[idx])
        
        ##-----------video feature-----------
        video = torch.load(self.video_path[idx])
        # (3x16x312x312) random crop(3x16x224x224)
        if self.is_train == True:
            h_start = random.randint(0, 88)
            h_end = h_start + 224
            w_start = random.randint(0, 88)
            w_end = w_start + 224
        else:
            h_start = 44
            h_end = h_start + 224
            w_start = 44
            w_end = w_start + 224
        data['video'] = video[:, :, h_start:h_end, w_start:w_end]   
        
        ## ----------label to id-------------- 
        data['labels'] = torch.tensor(eval(self.labels[idx]))
        
        ## ----------video name---------------
        if self.is_train == False:
            data['video_path'] = self.text_path[idx].split("/")[-1].split(".")[0] + '.mp4' # 返回.mp4文件名
            
        return data

    def __len__(self):
        return len(self.video_path)