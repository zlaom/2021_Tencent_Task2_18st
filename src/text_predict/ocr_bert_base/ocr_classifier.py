import json
import torch
import torch.nn as nn
import numpy as np
import os
import time
import csv

from transformers import BertTokenizer, AdamW
from tqdm import tqdm

from models.my_bert import MyBertClassifier
from utils.gap_score import get_tag_id_dict, parse_input_json, parse_gt_json, calculate_gap
from utils.utils import id_to_lable_to_id, samples_per_cls
from dataloader.my_dataloader import my_dataloader

dataset_root = '../../../'

label_id_dic, id_label_dic = id_to_lable_to_id(dataset_root+"dataset/label_id.txt")
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# get feature
def get_features(config, val_loader, tokenizer, model):
    model.eval()
    # 输出测试的.json文件
    features = np.array([])
    with torch.no_grad():
        for batch in tqdm(val_loader, ncols=100):
            tokens = tokenizer(batch['ocr'], truncation=True,
                            padding=True, max_length=512, return_tensors="pt")

            input_ids = tokens['input_ids'].to(device)
            attention_mask = tokens['attention_mask'].to(device)

            # 计算分数及标签 + 排序
            pred = model(input_ids, attention_mask)
            features=np.append(features, model.featuremap.cpu().numpy())
#             break
    print(features)
    return features.reshape(-1,1024)


# eval process
def evaluate(config, val_loader, tokenizer, model):
    model.eval()
    # 输出测试的.json文件
    output = {}
    with torch.no_grad():
        for batch in tqdm(val_loader, ncols=100):
            tokens = tokenizer(batch['ocr'], truncation=True,
                            padding=True, max_length=512, return_tensors="pt")

            input_ids = tokens['input_ids'].to(device)
            attention_mask = tokens['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # 计算分数及标签 + 排序
            pred = model(input_ids, attention_mask)
            scores = torch.sigmoid(pred)[0]  # 数字转换为0-1,且去掉batch维度
            scores_sort = scores.sort(descending=True)  # 对score排序
            labels = [id_label_dic[i.item()]
                      for i in scores_sort.indices]  # 生成排序好的labels
            scores = scores_sort.values  # 排序好的scores
            # 保存输出项目到output
            one_output = {}
            mp4_path = batch['video_path'][0]
            one_output["result"] = [
                {"labels": labels[:82], "scores": ["%.4f" % scores[i] for i in range(82)]}]
            output[mp4_path] = one_output
    pred_json = './val_temp.json'  # 上面输出的结果文件
    # 输出.json结果文件
    with open(pred_json, 'w', encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=4)

    # 计算GAP
    tag_id_file = dataset_root+'dataset/label_id.txt'  # tag-id文件
    gt_json = dataset_root+'dataset/structuring/GroundTruth/train5k.txt'  # 赛道一所有数据标签文件

    tag_dict = get_tag_id_dict(tag_id_file)  # tage:id字典

    # video-[tag_id-score数组]数组
    pred_dict = parse_input_json(pred_json, tag_dict)
    gt_dict = parse_gt_json(gt_json, tag_dict)  # video-[tag_id-score数组]数组

    preds, labels = [], []
    for k in pred_dict:
        preds.append(pred_dict[k])
        labels.append(gt_dict[k])

    preds = np.stack(preds)  # stack有啥用?
    labels = np.stack(labels)
    gap = calculate_gap(preds, labels, top_k=20)

    model.train()
    return gap

# train process
def train(run_i, model, loss_fn, config,  train_loader, val_loader, root_path):
    model = nn.DataParallel(model)
    model.to(device)
    model.train()
    optimizer = AdamW(model.parameters(), lr=1e-5)
    result_path = root_path+"run{}/".format(run_i)
    # 创建训练结果保存文件夹
    os.makedirs(result_path, exist_ok=True)
    tokenizer = BertTokenizer.from_pretrained(config['model_name'])
    
    with open(root_path+'res.csv', 'a+') as out:
        writer = csv.writer(out)
        writer.writerow(["run","model_name", "epoch", "gap", "av_loss"])
    s = str(config)
    fw = open(result_path+"config.txt", 'w')    
    fw.write(s)
    best_gap = 0
    count = 0
    for epoch in range(50):
        loss_sum = 0
        num = 1
        with tqdm(train_loader, ncols=120) as t:
            for batch in t:
                optimizer.zero_grad()
                tokens = tokenizer(batch['ocr'], truncation=True,
                                padding=True, max_length=512, return_tensors="pt")

                input_id = tokens['input_ids'].to(device)
                attention_mask = tokens['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                pred = model(input_id, attention_mask)
                loss = loss_fn(pred, labels)
                loss_sum+=loss.item()
                t.set_description("Epoch %i"%(epoch+1))
                t.set_postfix(loss=loss.item(), av_loss=loss_sum/num)  # 显示loss
                num+=1
                loss.backward()
                optimizer.step()
        gap = evaluate(config, val_loader, tokenizer, model)
        with open(root_path+'res.csv', 'a+') as out:
            writer = csv.writer(out)
            writer.writerow([run_i, config["model_name"],epoch, gap, loss_sum/num])
        if(gap>best_gap):
            best_gap = gap
            count = 0
            model_path = result_path + "best.pth"
            torch.save(model.module.state_dict(), model_path)
            print("Saved best model to: "+ model_path)
        else:
            count = count+1
        model_path = result_path + "last.pth"
        torch.save(model.module.state_dict(), model_path)
        print("Saved last model to: "+ model_path)
        
        if(count>3):
            print("overfit stop train!")
            break

# pre process
def predict(config, model,checkpoint,predict_loader,reslut_path):
    
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    tokenizer = BertTokenizer.from_pretrained(config['model_name'])
    # 输出测试的.json文件
    output = {}
    with torch.no_grad():
        for batch in tqdm(predict_loader, ncols=100):

            tokens = tokenizer(batch['ocr'], truncation=True,
                            padding=True, max_length=512, return_tensors="pt")

            input_ids = tokens['input_ids'].to(device)
            attention_mask = tokens['attention_mask'].to(device)
            
            # 计算分数及标签 + 排序
            pred = model(input_ids, attention_mask)
            scores = torch.sigmoid(pred)[0] # 数字转换为0-1,且去掉batch维度
            scores_sort = scores.sort(descending=True) # 对score排序
            labels = [ id_label_dic[i.item()] for i in scores_sort.indices ] # 生成排序好的labels
            scores = scores_sort.values # 排序好的scores
            
            # 保存输出项目到output
            one_output = {}
            one_output["result"] = [{"labels": labels[:], "scores": ["%.4f" % scores[i] for i in range(82)]}]
            mp4_path = batch['video_path'][0]
            output[mp4_path] = one_output

    # 输出.json测试文件
    with open(reslut_path, 'w', encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent = 4)
        
def run():
    train_sign = 'final_train'
    config = {
        'model_name':'hfl/chinese-macbert-large',
        'dropout':0.1,
        'learning_rate':1e-5,
#         'weight_decay':1e-2,
        'hidden_dim': 1024,
        'output_dim':82,
        'batch_size':10
    }

    
    # train
    print("============ begin train ==============")
    
    train_path = dataset_root+'dataset/tagging/GroundTruth/datafile/train.txt'
    val_path = dataset_root+'dataset/tagging/GroundTruth/datafile/val.txt'
    val_loader = my_dataloader(val_path, label_id_dic, False, batch_size=1, shuffle=True)
    train_loader = my_dataloader(train_path, label_id_dic, False, batch_size=config["batch_size"], shuffle=True)
    model = MyBertClassifier(config)
    loss = nn.BCEWithLogitsLoss()
    train(0, model, loss, config, train_loader, val_loader, "./checkpoints/{}/".format(train_sign))
    
    print("============ end train ================")
    
    # predict val
    print("============ begin predict val ==============")
    
    file_path = dataset_root+'dataset/tagging/GroundTruth/datafile/val.txt'
    checkpoint = torch.load("./checkpoints/{}/run0/best.pth".format(train_sign))
    model = MyBertClassifier(config)
    predict_loader = my_dataloader(file_path, label_id_dic, False, batch_size=1, shuffle=False)
    predict(config, model, checkpoint, predict_loader,"./result/val_large_ocr_val.json")
    
    print("============ end predict val ==============")
    
    # predict test
#     print("============ begin predict test ==============")

#     file_path = dataset_root+'test_2nd.txt'
#     checkpoint = torch.load("./checkpoints/{}/run0/best.pth".format(train_sign))
#     model = MyBertClassifier(config)
#     predict_loader = my_dataloader(file_path,label_id_dic, True, batch_size=1, shuffle=False)
#     predict(config, model, checkpoint, predict_loader,"./result/test_large_ocr_val.json")
    
#     print("============ end predict test ==============")
    
    
run()
    
# if __name__ == "__main__":
#     # Training on cuda
#     print(device)
#     train_sign = 'ocr_base_large_test'
#     action = 4
#     configs = [{
#         'model_name':'hfl/chinese-macbert-base',
#         'dropout':0.1,
#         'learning_rate':1e-5,
#         'weight_decay':1e-2,
#         'hidden_dim': 768,
#         'output_dim':82,
#         'batch_size':32
#     },{
#         'model_name':'hfl/chinese-macbert-large',
#         'dropout':0.1,
#         'learning_rate':1e-5,
#         'weight_decay':1e-2,
#         'hidden_dim': 1024,
#         'output_dim':82,
#         'batch_size':10
#     }]
    
#     if(action == 0):
#         print("begin train")
#         train_path = '../../dataset/tagging/GroundTruth/datafile/train.txt'
#         val_path = '../../dataset/tagging/GroundTruth/datafile/val.txt'
#         file_path = "train.txt"
#         # samples
#         with open(file_path, 'r') as f:
#             train_data = f.read().splitlines() # path_file是一个list
#             samples = samples_per_cls(train_data, label_id_dic)
#             print(samples)
#         # dataloader
#         val_loader = my_dataloader(val_path, label_id_dic, False, batch_size=1, shuffle=True) # here batch_size must be 1
#         for i, config in enumerate(configs):
#             print("Start train test:{}".format(i+1))
#             if(i in [0]):
#                 print("done! continue...")
#                 continue
#             print(config)
#             train_loader = my_dataloader(train_path, label_id_dic, False, batch_size=config["batch_size"], shuffle=True)
#             model = MyBertClassifier(config)
# #             loss = ClassBalanceLoss(samples, config['beta'], device)
#             loss = nn.BCEWithLogitsLoss()
#             train(i, model, loss, config, train_loader, val_loader, "./checkpoints/{}/".format(train_sign))
#     elif(action == 1):
#         print("begin predict")
#         config = configs[1]
#         test_path = './test_2nd.txt'
#         checkpoint = torch.load("./checkpoints/ocr_base_large_test/run1/best.pth")
#         model = MyBertClassifier(config)
#         predict_loader = my_dataloader(test_path,label_id_dic, True, batch_size=1, shuffle=False)
#         predict(config, model, checkpoint, predict_loader,"./test_large_ocr_val0.741.json")
#     elif(action == 2):
#         print("begin eval")
#         val_path = '../../dataset/tagging/GroundTruth/datafile/val.txt'
#         checkpoint = torch.load("./checkpoints/ocr_base_large_test/run1/best.pth")
#         val_loader = my_dataloader(val_path, label_id_dic, False, batch_size=1, shuffle=False) # here batch_size must be 1
#         config = configs[1]
#         model = MyBertClassifier(config)
#         model.load_state_dict(checkpoint)
#         model.to(device)
#         tokenizer = BertTokenizer.from_pretrained(config['model_name'])
#         gap = evaluate(config, val_loader, tokenizer, model)
#         print('gap:{}'.format(gap))
#     elif action == 3:
#         print('get features')
#         val_path = './test.txt'
#         checkpoint = torch.load("./checkpoints/ocr_base_large_test/run1/best.pth")
#         val_loader = my_dataloader(val_path, label_id_dic, True, batch_size=1, shuffle=False) # here batch_size must be 1
#         config = configs[1]
#         model = MyBertClassifier(config)
#         model.load_state_dict(checkpoint)
#         model.to(device)
#         tokenizer = BertTokenizer.from_pretrained(config['model_name'])
#         features = get_features(config, val_loader, tokenizer, model)
#         print(features.shape)
#         np.save('test_features.npy', features)
#     elif action == 4:
#         print('evaluate')
#         val_path = '../../dataset/tagging/GroundTruth/datafile/train.txt'
#         checkpoint = torch.load("./checkpoints/ocr_base_large_test/run1/best.pth")
#         val_loader = my_dataloader(val_path, label_id_dic, False, batch_size=1, shuffle=False) # here batch_size must be 1
#         config = configs[1]
#         model = MyBertClassifier(config)
#         model.load_state_dict(checkpoint)
#         model.to(device)
#         tokenizer = BertTokenizer.from_pretrained(config['model_name'])
#         gap = evaluate(config, val_loader, tokenizer, model)
#         print('gap:', gap)

        