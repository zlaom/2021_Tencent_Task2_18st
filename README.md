# 2021_Tencent_Task2_18st

2021腾讯广告算法大赛 赛道二 多模态视频标签预测任务 第18名方案

## 整体思路

通过NeXtVLAD对音频和视频进行预测和融合，验证集效果为gap0.771

通过X3D对视频进行单独预测，验证集效果为gap0.765

通过BERT对文本进行单独预测，验证集效果为gap0.742
随后根据0.742 BERT结果对准确率进行划分后训练预测得验证集结果为gap0.7456

融合过程：首先将X3D与文本融合得验证集gap0.785，再将X3D与NeXtVLAD融合得gap0.7862，融合方式为提取最后的隐藏层作为特征进行交叉SeNet融合。最后再将两个预测结果及文本的预测结果进行ensemble融合得验证集0.8086。

## 环境初始化

修改init.sh中dataset所在目录

```bash
# #################### link dataset
DATASET_ROOT=/home/tione/notebook/algo-2021/dataset/
```

执行初始化

`sudo chmod a+x ./init.sh && ./init.sh run`

## 训练

我们在网盘提供了NeXtVLAD的自提特征，由于提特征需要很长时间，而且一站式代码并未实现并行处理，如果自己运行提特征可能需要单卡几天的时间。强烈推荐在网盘下载，将文件解压到/home/tione/notebook/dataset/ 文件夹下.

```txt
# 放置特征后目录如下
dataset
    - features
        - train
        - val
        - test
```

特征下载路链接：<http://vtrust.qicp.vip:45124/s/8g5RFBgTpsN7pEe>

备用链接：<https://pan.baidu.com/s/1yiSiXruG-j7EWIX75vTjpA> 提取码：6ylh

请在根目录下运行如下代码
`sudo chmod a+x ./run.sh && ./run.sh train`

如果要运行完整的代码（包括NeXtVLAD提特征），运行
`sudo chmod a+x ./run.sh && ./run.sh train_plus`

## 测试

`sudo chmod a+x ./run.sh && ./run.sh test`
