# 多模态标签分类预测

## 整体思路

通过nextvlad对音频和视频进行预测和融合，验证集效果在0.771

通过x3d对视频进行单独预测，验证集效果在0.765

通过bert对文本进行单独预测，验证集效果在0.742

融合过程：首先将x3d与文本融合，再将x3d与nextvlad融合，融合方式为提取最后的隐藏层作为特征进行交叉senet融合。最后再将两个预测结果及文本的预测结果进行ensemble融合。

## 环境初始化

修改init.sh中dataset所在目录

```
# #################### link dataset
DATASET_ROOT=/home/tione/notebook/algo-2021/dataset/
```

执行初始化

`sudo chmod a+x ./init.sh && ./init.sh run`

## 训练
我们在网盘提供了nextvlad的自提特征，由于提特征需要很长时间，而且一站式代码并未实现并行处理，如果自己运行提特征可能需要单卡几天的时间。强烈推荐在网盘下载，将文件解压到/home/tione/notebook/dataset/ 文件夹下.

```
# 放置特征后目录如下
dataset
    - features
        - train
        - val
        - test
```

下载路径为：http://vtrust.qicp.vip:45124/s/8g5RFBgTpsN7pEe

请在根目录下运行如下代码
`sudo chmod a+x ./run.sh && ./run.sh train`

如果要运行完整的代码（包括nextvlad提特征），运行
`sudo chmod a+x ./run.sh && ./run.sh train_plus`

## 测试
`sudo chmod a+x ./run.sh && ./run.sh test`