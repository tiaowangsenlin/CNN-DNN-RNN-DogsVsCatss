from getdata import DogsVSCatsDataset as DVCD
from CNN.CNN_network import Net
import torch
import torch.nn.functional as F
import torch.nn as nn

dataset_dir = '../data'  # 数据集路径
model_file = '../model/cnn_model.pth'  # 模型保存路径


# old version
def test():

    model = Net()                                       # 实例化一个网络
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(model_file))       # 加载训练好的模型参数
    model.eval()                                        # 设定为评估模式，即计算过程中不要dropout

    datafile = DVCD('test', dataset_dir)                # 实例化一个数据集
    print('Dataset loaded! length of train set is {0}'.format(len(datafile)))
    count = 0
    N = datafile.data_size
    acc = 0
    for index in range(N):
        img = datafile.__getitem__(index)                           # 获取一个图像
        img = img.unsqueeze(0)                                      # 因为网络的输入是一个4维Tensor，3维数据，1维样本大小，所以直接获取的图像数据需要增加1个维度
        lable = datafile.list_label[index]                          # 获取test集合中的图片标签，0：猫，1：狗

        out = model(img)                                            # 网路前向计算，输出图片属于猫或狗的概率，第一列维猫的概率，第二列为狗的概率
        out = F.softmax(out, dim=1)                                 # 采用SoftMax方法将输出的2个输出值调整至[0.0, 1.0],两者和为1
        print(out)                                                  # 输出该图像属于猫或狗的概率
        if out[0, 0] > out[0, 1] :                                  # 猫的概率大于狗
            print('the image is a cat and lable is',lable)
        else:                                                       # 猫的概率小于狗
            print('the image is a dog and lable is',lable)

        if (out[0, 0] > out[0, 1] and lable == 0 ) or (out[0, 0] <= out[0, 1] and lable==1): #记录识别正确的数量
            count = count+1

    acc = count/N
    print('准确率:', acc)


if __name__ == '__main__':
    test()


