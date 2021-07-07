import os
import torch
import torch.nn as nn
import numpy as np
import json

from torch.optim import AdamW
from tqdm import tqdm
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_normal_, uniform_
from torch.utils.data import DataLoader

from utils.utils import id_to_lable_to_id, samples_per_cls
from utils.gap_score import parse_input_json, parse_gt_json, calculate_gap
label_id_dic, id_label_dic = id_to_lable_to_id("./utils/label_id.txt")
gt_json = 'utils/train5k.txt'

def data_transform(f_paths, is_test):
    origin = [parse_input_json(f_path, label_id_dic) for f_path in f_paths]
    video_name = list(origin[0].keys())
    label = []
    new = []
    for item in origin:
        one = []
        for key in video_name:
            one.append(item[key])          
        new.append(one)
    
    if is_test:
        return new, video_name
    else:
        gt_dict = parse_gt_json(gt_json, label_id_dic)
        for key in video_name:
            label.append(gt_dict[key])
        return new, video_name, label
    
# 训练权重因子
class WeightClassifier1(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.alph = Parameter(torch.tensor([config['beta']]*82))
    def forward(self, f1, f2):
        output = torch.mul(f1, self.alph)+torch.mul(f2, 1-self.alph)
        return output
    
# class WeightClassifier2(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.alph = Parameter(torch.full(( config['featrues_num'], config['output_dim']), 1/config['featrues_num']))
#         uniform_(self.alph, a=0.4, b=0.6)
#     def forward(self, f1, f2):
#         input_data =  torch.cat((torch.unsqueeze(f1, 1),torch.unsqueeze(f2, 1)), axis = 1)
#         imput_data = torch.mul(self.alph, input_data)
#         output = torch.sigmoid(torch.sum(imput_data, axis=1))
#         return output

# dataset
class WeightDataset(torch.utils.data.Dataset):
    def __init__(self, data, is_test):
        self.is_test = is_test
        self.f1 = data['f1']
        self.f2 = data['f2']
        self.name = data['f3']
        if(not is_test):
            self.label = data['f4']
            
    def __getitem__(self, idx):
        data = {}
        data['f1'] = self.f1[idx]
        data['f2'] = self.f2[idx]
        data['name'] = self.name[idx]
        
        if(not self.is_test):
            data['label'] = self.label[idx]
            
        return data
    def __len__(self):
        return len(self.f1)
    
# dataloader
def my_dataloader(data, is_test, batch_size, shuffle):
    dataset = WeightDataset(data, is_test)
    return DataLoader(dataset = dataset, batch_size = batch_size, shuffle = shuffle)

## calculate gap
def get_gap(pred_dict, gt_dict, k = 20):
    preds, labels = [], []
    for i in pred_dict:
        preds.append(pred_dict[i])
        labels.append(gt_dict[i])
    preds = np.stack(preds)
    labels = np.stack(labels)
    return calculate_gap(preds, labels, top_k=k)

def my_get_gap(pred_json):
    pred_dict = parse_input_json(pred_json, label_id_dic)
    gt_dict = parse_gt_json(gt_json, label_id_dic)
    return get_gap(pred_dict, gt_dict)

# evalue
def evalue(model, config):
    model.eval()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    output = {}
    for batch in config['eval_loader']:
        f1 = batch['f1'].to(device).float()
        f2 = batch['f2'].to(device).float()
        pred = model(f1, f2)
        scores = torch.sigmoid(pred)[0]  # 数字转换为0-1,且去掉batch维度
        scores_sort = scores.sort(descending=True)  # 对score排序
        labels = [id_label_dic[i.item()]
        for i in scores_sort.indices]  # 生成排序好的labels
        scores = scores_sort.values  # 排序好的scores
        one_output = {}
        mp4_path = batch['name'][0]
        one_output["result"] = [{"labels": labels[:config['scope']], "scores": ["%.4f" % scores[i] for i in range(config['scope'])]}]
        output[mp4_path] = one_output
    pred_json = 'result/{}_result.json'.format(config['train_save_name'])
    if(config['state']=='train'):
        pred_json = 'result/{}_temp_result.json'.format(config['train_save_name'])
    
    # 输出.json结果文件
    with open(pred_json, 'w', encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=4)
    model.train()
    gap = my_get_gap(pred_json)
    return gap

# train
def train(config):
    print('begin trian ...')
    model = WeightClassifier1(config)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.train()
    
    optimizer = AdamW(model.parameters(), lr=config['lr'])
    loss_fn = nn.BCEWithLogitsLoss()

    best_gap = 0
    best_param = []
    flag = 0
    for epoch in range(config['epoch']):
        for batch in config['train_loader']:
            optimizer.zero_grad()
            f1 = batch['f1'].to(device).float()
            f2 = batch['f2'].to(device).float()
            label = batch['label'].to(device)
            pred = model(f1, f2)
            loss = loss_fn(pred, label)

            loss.backward()
            optimizer.step()
        gap = evalue(model, config)

        
        if best_gap < gap:
            flag = 0
            for name, param in model.named_parameters():
                if (name in ['alph']) :
                    best_gap = gap
                    best_param = param.detach().to('cpu').numpy()
                    np.save(config['train_save_name']+'_best_w', best_param)
                    torch.save(model.state_dict(), config['train_save_name']+'_best_model.pth')
        else:
            flag += 1
        
        print('gap', gap, best_gap, flag)
        if(flag>config['early_stop']):break
    print('end train .')
    return best_gap, best_param

# predict
def predict(config, f1, f2, w, video_name):
    result = 1/(1+np.exp(-(f1*w+f2*(1.0-w))))
    output = {}
    for idx, name in enumerate(video_name):
        scores = result[idx]
        scores_sort = torch.tensor(scores).sort(descending=True)
        labels = [ id_label_dic[i.item()] for i in scores_sort.indices ] # 生成排序好的labels
        scores = scores_sort.values # 排序好的scores
        one_output = {}
        one_output["result"] = [{"labels": labels[:config['scope']], "scores": ["%.4f" % scores[i] for i in range(config['scope'])]}]
        output[name] = one_output
    return result, output


def run_train(config, datas):
    new_datas, video_name, label = data_transform(datas, is_test=False)
    loader_data = {
        'f1':new_datas[0],
        'f2':new_datas[1],
        'f3':video_name,
        'f4':label
    }

    config['train_loader'] = my_dataloader(loader_data, False, 32, True)
    config['eval_loader'] = my_dataloader(loader_data, False, 1, False)

    beat_gap, best_param = train(config)
    print(beat_gap, best_param)
    
    result, output = predict(config, new_datas[0], new_datas[1], best_param, video_name)
    np.save('result/{}'.format(config['save_name']), best_param)
    pred_json = 'result/{}.json'.format(config['save_name'])
    # 输出.json结果文件
    with open(pred_json, 'w', encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=4)

config = {
    'state':'train',
    'epoch':300,
    'output_dim':82,
    'featrues_num':2,
    'beta':0.59,
    'lr':1e-3,
    'run':1,
    'scope':20,
    'train_save_name':'temp_train',
    'save_name':'',
    'early_stop':20
}

# print('================== start weight fusion train =================')

# x3d_nextvlad = 'dataset/x3d_nextvlad_val.json'
# class_bert = 'dataset/class_bert_val.json'
# x3d_bert = 'dataset/x3d_bert_val.json'

# x3d_nextvlad_and_class_bert = 'result/x3d_nextvlad_and_class_bert_val.json'


# # 0.5-8076 
# config['save_name']='x3d_nextvlad_and_class_bert_val'
# config['beta']=0.5
# config['lr']=5e-4
# datas = [
#     x3d_nextvlad,
#     class_bert,
# ]

# run_train(config, datas)

# # 1-80862
# config['save_name']='x3d_nextvlad_and_class_bert_and_x3d_bert_val'
# config['beta']=1.0
# config['lr']=5e-5
# datas = [
#     x3d_nextvlad_and_class_bert,
#     x3d_bert,
# ]

# run_train(config, datas)

# print('================== end weight fusion train =================')



print('================== start weight fusion predict =================')

x3d_nextvlad = 'dataset/x3d_nextvlad_test.json' #change
class_bert = '../text_predict/ocr_bert_class_train/result/test.json' # change
x3d_bert = 'dataset/x3d_bert_test.json' #change

x3d_nextvlad_and_class_bert = 'result/x3d_nextvlad_and_class_bert_test.json'

# se x3d nextvlad + bert
datas = [
    x3d_nextvlad,
    class_bert
]

new_datas, video_name = data_transform(datas, is_test=True)
loader_data = {
    'f1':new_datas[0],
    'f2':new_datas[1],
    'f3':video_name,
    'f4':''
}

best_param = np.load('result/x3d_nextvlad_and_class_bert_val.npy')
config['eval_loader'] = my_dataloader(loader_data, True, 1, False)
result, output = predict(config, new_datas[0], new_datas[1], best_param, video_name)

pred_json = 'result/x3d_nextvlad_and_class_bert_test.json'
# 输出.json结果文件
with open(pred_json, 'w', encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=4)
print("finish save to {}".format(pred_json))

# (se x3d nextvlad + bertx3d) + (se x3d bert)
datas = [
    x3d_nextvlad_and_class_bert,
    x3d_bert
]

new_datas, video_name = data_transform(datas, is_test=True)
loader_data = {
    'f1':new_datas[0],
    'f2':new_datas[1],
    'f3':video_name,
    'f4':''
}

best_param = np.load('result/x3d_nextvlad_and_class_bert_and_x3d_bert_val.npy')
config['eval_loader'] = my_dataloader(loader_data, True, 1, False)
result, output = predict(config, new_datas[0], new_datas[1], best_param, video_name)

pred_json = 'result/x3d_nextvlad_and_class_bert_and_x3d_bert_test.json'
# 输出.json结果文件
with open(pred_json, 'w', encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=4)
print("finish save to {}".format(pred_json))

print('================== end weight fusion predict =================')

print('\n\nFinal result is: {}/{}'.format(os.getcwd(),pred_json))