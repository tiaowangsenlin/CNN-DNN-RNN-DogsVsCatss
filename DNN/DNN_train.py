import os

from matplotlib import pyplot as plt
from torch.utils.data import DataLoader as DataLoader
import torch
from getdata import DogsVSCatsDataset as DVCD
from DNN.DNN_network import myNet
from torch.autograd import Variable
import torch.nn as nn

dataset_dir = '../data/'  # 数据集路径
model_cp = './model/'  # 网络模型保存位置
batch_size = 100  # batch_size大小，每一批次训练
lr = 0.0001  # 学习率
nepoch = 10  # 整个数据集训练的次数，train中共4000张图片，训练20次


def train():
    datafile = DVCD('train', dataset_dir)  # 实例化一个数据集
    dataloader = DataLoader(datafile, batch_size=batch_size, shuffle=True)
    print('Dataset loaded! length of train set is {0}'.format(len(datafile)))

    model = myNet()
    model = nn.DataParallel(model)
    model.train()  # 网络设定为训练模式，有两种模式可选，.train()和.eval()，训练模式和评估模式，区别就是训练模式采用了dropout策略，可以放置网络过拟合
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # 实例化一个优化器，即调整网络参数，优化方式为adam方法
    criterion = torch.nn.CrossEntropyLoss()
    cnt = 0
    loss_count = []
    for epoch in range(nepoch):  # 所有的训练集训练十次
        for img, label in dataloader:
            img, label = Variable(img), Variable(label)
            out = model(img)

            loss = criterion(out, label.squeeze())
            loss.backward()  # 误差反向传播，采用求导的方式，计算网络中每个节点参数的梯度，显然梯度越大说明参数设置不合理，需要调整

            optimizer.step()  # 优化采用设定的优化方法对网络中的各个参数进行调整
            optimizer.zero_grad()  # 清除优化器中的梯度以便下一次计算，因为优化器默认会保留，不清除的话，每次计算梯度都回累加

            cnt += 1
            if cnt % 10 == 0:
                loss_count.append(loss)
            print('Epoch:{0},Frame:{1}, train_loss {2}'.format(epoch, cnt * batch_size, loss / batch_size))

    torch.save(model, os.path.join('../model/', 'dnn_model.pt'))
    plt.figure('DNN_loss')
    plt.plot(loss_count, label='loss')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    train()
