import torch
import torch.nn as nn
import json
import numpy as np
import os
import time
import csv

from transformers import BertTokenizer, AdamW
from tqdm import tqdm
from transformers import AutoModel
from torch.utils.data import DataLoader
from utils.gap_score import get_tag_id_dict, parse_input_json, parse_gt_json, calculate_gap
from utils.utils import id_to_lable_to_id

dataset_root = '../../../'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

tag_id_file = dataset_root+'dataset/label_id.txt'  # tag-id文件
label_id_dic, id_label_dic = id_to_lable_to_id(tag_id_file)

class MyBertClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert = AutoModel.from_pretrained(config['model_name'])
        self.classifier = nn.Sequential(
            nn.Dropout(config['dropout']),
            nn.Linear(config['hidden_dim'], config['output_dim'])
        )
    def forward(self, input_id, attention_mask):
        x = self.bert(input_id, attention_mask).pooler_output
        self.featuremap = x # 核心代码
        x = self.classifier(x)
        return x
    
class MaskTextDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, mask, is_test):
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
                ids[label_id_dic[label]] = 1
            data['labels'] = ids[self.mask]
        return data

    def __len__(self):
        return len(self.video_path)

def dataloader(file_path, mask, is_test, batch_size, shuffle):
    dataset = MaskTextDataset(file_path, mask, is_test)
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)

def train(config, model, data):
    model = nn.DataParallel(model)
    model.to(device)
    model.train()
    
    result_path = "checkpoints/run{}/".format(config['run'])
    config['result_path'] = result_path
    os.makedirs(result_path, exist_ok=True)
    
    with open(result_path+'res.csv', 'w') as out:
        writer = csv.writer(out)
        writer.writerow(["run","model_name", "epoch",  "av_loss", "run_time", "epoch gap", "best gap"])
        
    s = str(config)
    fw = open(result_path+"config.txt", 'w')    
    fw.write(s)
    best_gap = 0
    count = 0
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=config['lr'])
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    
    for epoch in range(config['epoch']):
        epoch_best_gap = 0
        loss_sum = 0
        num = 1
        start = time.clock()
        with tqdm(data['train_loader'], ncols=120) as t:
            for idex, batch in enumerate(t):
                optimizer.zero_grad()
                tokens = tokenizer(batch['ocr'], truncation=True,
                                padding=True, max_length=512, return_tensors="pt")

                input_id = tokens['input_ids'].to(device)
                attention_mask = tokens['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                pred = model(input_id, attention_mask)
                loss = loss_fn(pred, labels)
                loss_sum+=loss.item()

                num+=1
                loss.backward()
                optimizer.step()
                if idex%100==0:
                    gap = evaluate(config, model, data)
                    
                    if(gap>epoch_best_gap):
                        epoch_best_gap=gap
                        
                    if(epoch_best_gap>best_gap):
                        best_gap = epoch_best_gap
                        
                        model_path = result_path + "{}_best.pth".format(config['model_tag'])
                        torch.save(model.module.state_dict(), model_path)
                        print("\nSaved best model to: "+model_path)

                    t.set_description("Epoch %i"%(epoch+1))
                    t.set_postfix(av_loss=loss_sum/num, gap=gap, best_gap=best_gap, count = count)  # 显示loss
#                     scheduler.step()

        model_path = result_path + "{}_last.pth".format(config['model_tag'])
        torch.save(model.module.state_dict(), model_path)
        print("\nSaved last model to: "+model_path)

        end = time.clock()
        print('Running time: %s Seconds'%(end-start))
        if(epoch_best_gap>=best_gap):
            count=0
        else:
            count+=1
        with open(result_path+'res.csv', 'a+') as out:
            writer = csv.writer(out)
            writer.writerow([config['run'], config["model_name"], epoch, loss_sum/num, end-start, epoch_best_gap, best_gap])
#         if(gap>best_gap):
#             best_gap = gap
#             count = 0
            
#         else:
#             count = count+1

        
        if(count>config['early_stop']):
            print("overfit stop train!")
            break

def predict(config, model, data):
    model.to(device)
    model.eval()
    output = {}
    result_path = "checkpoints/run{}/".format(config['run'])
    config['result_path'] = result_path
#     print(data, data['val_loader'])
    with torch.no_grad():
        for batch in tqdm(data['val_loader'], ncols=100):
            tokens = tokenizer(batch['ocr'], truncation=True,
                            padding=True, max_length=512, return_tensors="pt")

            input_ids = tokens['input_ids'].to(device)
            attention_mask = tokens['attention_mask'].to(device)
#             labels = batch['labels'].to(device)

            # 计算分数及标签 + 排序
            pred = model(input_ids, attention_mask)
            scores = torch.sigmoid(pred)[0]  # 数字转换为0-1,且去掉batch维度
            
            
            scores_np = np.zeros(82)
#             print( scores.cpu().numpy())
#             print(data['mask_index'])
            scores_np[data['mask_index'].reshape((-1,))] = scores.cpu().numpy()
            scores_index = np.argsort(scores_np)[::-1]
            scores_np = scores_np[scores_index]
            
            labels = [id_label_dic[i.item()]
                      for i in scores_index]  # 生成排序好的labels
            scores = scores_np  # 排序好的scores
            # 保存输出项目到output
            one_output = {}
            mp4_path = batch['video_path'][0]
            one_output["result"] = [
                {"labels": labels[:82], "scores": ["%.4f" % scores[i] for i in range(82)]}]
            output[mp4_path] = one_output
    pred_json = config['result_path']+'val.json'  # 上面输出的结果文件
    if config['state']=='test':
        pred_json = config['result_path']+'test.json'  # 上面输出的结果文件
   
    # 输出.json结果文件
    with open(pred_json, 'w', encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=4) 

# def evaluate(config, model, data):
#     model.eval()
#     output = {}
#     with torch.no_grad():
#         for batch in tqdm(data['val_loader'], ncols=100):
#             tokens = tokenizer(batch['ocr'], truncation=True,
#                             padding=True, max_length=512, return_tensors="pt")

#             input_ids = tokens['input_ids'].to(device)
#             attention_mask = tokens['attention_mask'].to(device)
#             labels = batch['labels'].to(device)

#             # 计算分数及标签 + 排序
#             pred = model(input_ids, attention_mask)
#             scores = torch.sigmoid(pred)[0]  # 数字转换为0-1,且去掉batch维度
            
            
#             scores_np = np.zeros(82)
# #             print( scores.cpu().numpy())
# #             print(data['mask_index'])
#             scores_np[data['mask_index'].reshape((-1,))] = scores.cpu().numpy()
#             scores_index = np.argsort(scores_np)[::-1]
#             scores_np = scores_np[scores_index]
            
#             labels = [id_label_dic[i.item()]
#                       for i in scores_index]  # 生成排序好的labels
#             scores = scores_np  # 排序好的scores
#             # 保存输出项目到output
#             one_output = {}
#             mp4_path = batch['video_path'][0]
#             one_output["result"] = [
#                 {"labels": labels[:82], "scores": ["%.4f" % scores[i] for i in range(82)]}]
#             output[mp4_path] = one_output
    
#     pred_json = config['result_path']+'val.json'  # 上面输出的结果文件
#     # 输出.json结果文件
#     with open(pred_json, 'w', encoding="utf-8") as f:
#         json.dump(output, f, ensure_ascii=False, indent=4)        
#     tag_id_file = '../../dataset/label_id.txt'  # tag-id文件
#     gt_json = '../../dataset/structuring/GroundTruth/train5k.txt'  # 赛道一所有数据标签文件

#     # video-[tag_id-score数组]数组
#     pred_dict = parse_input_json(pred_json, label_id_dic)
#     gt_dict = parse_gt_json(gt_json, label_id_dic)  # video-[tag_id-score数组]数组

#     preds, labels = [], []
#     for k in pred_dict:
#         preds.append(pred_dict[k])
#         gt_ = np.zeros(82)
#         temp = np.array(gt_dict[k])[data['mask_index']]
#         gt_[data['mask_index']]=temp
# #         print(gt_)
#         labels.append(gt_)

#     preds = np.stack(preds)
#     labels = np.stack(labels)
#     gap = calculate_gap(preds, labels, top_k=20)

#     model.train()
#     return gap
def evaluate(config, model, data):
    model.eval()
    output = {}
    with torch.no_grad():
        for batch in data['val_loader']:
            tokens = tokenizer(batch['ocr'], truncation=True,
                            padding=True, max_length=512, return_tensors="pt")

            input_ids = tokens['input_ids'].to(device)
            attention_mask = tokens['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # 计算分数及标签 + 排序
            predicts = model(input_ids, attention_mask)
            for i, predict in enumerate(predicts):
                scores = torch.sigmoid(predict) # 数字转换为0-1,且去掉batch维度
                scores_np = np.zeros(82)
                scores_np[data['mask_index'].reshape((-1,))] = scores.cpu().numpy()
                scores_index = np.argsort(scores_np)[::-1]
                scores_np = scores_np[scores_index]

                labels = [id_label_dic[i.item()]
                          for i in scores_index]  # 生成排序好的labels
                scores = scores_np  # 排序好的scores
                # 保存输出项目到output
                one_output = {}
                mp4_path = batch['video_path'][i]
                one_output["result"] = [
                    {"labels": labels[:82], "scores": ["%.4f" % scores[i] for i in range(82)]}]
                output[mp4_path] = one_output
    pred_json = config['result_path']+'temp_val.json'  # 上面输出的结果文件
    # 输出.json结果文件
    with open(pred_json, 'w', encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=4)        
    tag_id_file = dataset_root+'dataset/label_id.txt'  # tag-id文件
    gt_json = dataset_root+'dataset/structuring/GroundTruth/train5k.txt'  # 赛道一所有数据标签文件

    # video-[tag_id-score数组]数组
    pred_dict = parse_input_json(pred_json, label_id_dic)
    gt_dict = parse_gt_json(gt_json, label_id_dic)  # video-[tag_id-score数组]数组

    preds, labels = [], []
    for k in pred_dict:
        preds.append(pred_dict[k])
        gt_ = np.zeros(82)
        temp = np.array(gt_dict[k])[data['mask_index']]
        gt_[data['mask_index']]=temp
        labels.append(gt_)
    preds = np.stack(preds)
    labels = np.stack(labels)
    gap = calculate_gap(preds, labels, top_k=20)
    
    model.train()
    return gap

def get_mask(run):
    text_result = np.load('./text_result.npy')
    mask = np.zeros(82)
    n = 0
    hard_class = []
    if run==1:
        hard_class = np.argwhere(text_result<0.5)
        n = hard_class.shape[0]
        mask[hard_class] = 1
    elif run==2:
        hard_class = np.argwhere(text_result>0.5)
        n = hard_class.shape[0]
        mask[hard_class] = 1
    elif run==3:
        hard_class = np.argwhere(text_result>0.1)
        n = hard_class.shape[0]
        mask[hard_class] = 1
    elif run==4:
        hard_class = np.argwhere(np.logical_and(text_result>0.1 , text_result<=0.9))
        n = hard_class.shape[0]
        mask[hard_class] = 1
    elif run==5:
        hard_class = np.argwhere(text_result>0.9)
        n = hard_class.shape[0]
        mask[hard_class] = 1
    elif run==6:
        hard_class = np.argwhere(text_result<=0.1)
        n = hard_class.shape[0]
        mask[hard_class] = 1
    elif run==7:
        hard_class = np.argwhere(np.logical_and(text_result>0.1 , text_result<=0.5))
        n = hard_class.shape[0]
        mask[hard_class] = 1
    elif run==8:
        hard_class = np.argwhere(np.logical_and(text_result>0.5 , text_result<=0.9))
        n = hard_class.shape[0]
        mask[hard_class] = 1
    elif run==9:
        hard_class = np.argwhere(np.logical_and(text_result>0.1 , text_result<=0.3))
        n = hard_class.shape[0]
        mask[hard_class] = 1
    elif run==10:
        hard_class = np.argwhere(np.logical_and(text_result>0.3 , text_result<=0.5))
        n = hard_class.shape[0]
        mask[hard_class] = 1
    elif run==11:
        hard_class = np.argwhere(np.logical_and(text_result>0.5 , text_result<=0.7))
        n = hard_class.shape[0]
        mask[hard_class] = 1
    elif run==12:
        hard_class = np.argwhere(np.logical_and(text_result>0.7 , text_result<=0.9))
        n = hard_class.shape[0]
        mask[hard_class] = 1
    elif run==13:
        hard_class = np.argwhere(text_result<=0.2)
        n = hard_class.shape[0]
        mask[hard_class] = 1
    elif run==14:
        hard_class = np.argwhere(np.logical_and(text_result>0.2 , text_result<=0.9))
        n = hard_class.shape[0]
        mask[hard_class] = 1
    elif run==15:
        hard_class = np.argwhere(text_result>0.9)
        n = hard_class.shape[0]
        mask[hard_class] = 1
    return hard_class, n, mask
    

def ini_predict(config, file_path, checkpoint):
    hard_class, n, mask =  get_mask(config['run'])
    config['output_dim'] = n # 修改
    
#     val_loader = dataloader(file_path, mask, True, batch_size=1, shuffle=False)
    val_loader = dataloader(file_path, mask, False, batch_size=1, shuffle=False)
    if config['state'] == 'test':
        val_loader = dataloader(file_path, mask, True, batch_size=1, shuffle=False)
    model = MyBertClassifier(config)
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint)
    data = {
        'val_loader':val_loader,
        'mask_index':hard_class
    }
    return model, data

def init_train(config):
    train_path = dataset_root+'dataset/tagging/GroundTruth/datafile/train.txt'
    val_path = dataset_root+'dataset/tagging/GroundTruth/datafile/val.txt'
    
    hard_class, n, mask =  get_mask(config['run'])
    
    config['output_dim'] = n 
    
    train_loader = dataloader(train_path, mask, False, batch_size=config['batch_size'], shuffle=True)
    val_loader = dataloader(val_path, mask, False, batch_size=100, shuffle=True)
    model = MyBertClassifier(config)

#     pretrained_dict=torch.load('./ocr_0.7403.pth')
#     model_dict=model.state_dict()
#     pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
#     print(pretrained_dict)
#     model_dict.update(pretrained_dict)
#     model.load_state_dict(model_dict)
    data = {
        'train_loader':train_loader,
        'val_loader':val_loader,
        'mask_index':hard_class
    }
    return model, data

def run():
    config = {
        'model_name':'hfl/chinese-macbert-large',
        'epoch':50,
        'dropout':0.1,
        'hidden_dim': 1024,
        'output_dim':82,
        'batch_size':8,
        'run':0,
        'state':'val',
        'lr':1e-5,
        'early_stop':2,
        'model_tag':'lr_scaler'
    }

if __name__ == "__main__":
    # 0-train 1-predict 2-
    action = 4
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    config = {
        'model_name':'hfl/chinese-macbert-large',
        'epoch':50,
        'dropout':0.1,
        'hidden_dim': 1024,
        'output_dim':82,
        'batch_size':8,
        'run':0,
        'state':'val',
        'lr':1e-5,
        'early_stop':2,
        'model_tag':'lr_scaler'
    }
    tokenizer = BertTokenizer.from_pretrained(config['model_name'])
    if action == 0:
        print("begin train")
        model, data = init_train(config)
        train(config, model, data)
        # dataloader
        # build model
        pass
    elif action == 1:
        print("begin predict")
#         file_path = 'test_2nd.txt'
        file_path = 'val.txt'
        checkpoint = 'checkpoints/run{}/best.pth'.format(config['run'])
        model, data = ini_predict(config, file_path, checkpoint)
        predict(config, model, data)
    elif action == 3:
        print("begin multiple training...")
        runs = [5,6,7,8]
        runs = [13]
        for run in runs:
            print('train run {}'.format(run))
            config['run'] = run
            model, data = init_train(config)
            train(config, model, data)
        print("end multiple training.")
    elif action == 4:
        print("begin multiple predict...")
        config['state']='val'
        runs = [5,6,9,10,11,12]
        runs = [4,5,6] #0.805438
        runs = [15,14,13]
        file_path = 'val.txt'
#         file_path = 'test_2nd.txt'
        for run in runs:
            print('predict run {}'.format(run))
            config['run'] = run
            checkpoint = 'checkpoints/run{}/{}_best.pth'.format(config['run'],config['model_tag'])
            model, data = ini_predict(config, file_path, checkpoint)
            predict(config, model, data)
        print("end multiple predict.")