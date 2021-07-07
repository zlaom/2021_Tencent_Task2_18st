import numpy as np
def total(params):
    '''
    count the total number of hyperparameter settings
    '''
    settings = 1
    for k, v in params.items():
        settings *= len(v)
    return settings
def id_to_lable_to_id(file_path):
    """
    获得标签转id以及id转标签字典
    """
    # 创建labels--ids转换字典
    label_id_dic = {}
    with open(file_path) as f:
        while True:
            line = f.readline()
            if line == '':
                break
            pair = line.strip().split('\t')
            label_id_dic[pair[0]] = int(pair[1])
    # 创建ids--labels转换字典，输出需要转换回来
    id_label_dic = {value: key for key, value in label_id_dic.items()}
    return label_id_dic, id_label_dic

# 统计标签数
def samples_per_cls(data, label_id_dic):
    labels = data[4::6]
    count = np.zeros(82, dtype=int)
    for label in labels:
        # print(label)
        label = label.split(',')
        for item in label:
            count[label_id_dic[item]] = count[label_id_dic[item]]+1
    return count

def label_nums(data):
    labels = data[4::6]
    count = []
    for label in labels:
        # print(label)
        label = label.split(',')
        count.append(len(label))
    return count

if __name__ == "__main__":
    label_id_dic, id_label_dic = id_to_lable_to_id("../../dataset/label_id.txt")
    file_path = "../../dataset/tagging/GroundTruth/datafile/train.txt"
    with open(file_path, 'r') as f:
            train_data = f.read().splitlines() # path_file是一个list
            samples = samples_per_cls(train_data, label_id_dic)
    print(samples)