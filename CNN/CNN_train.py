import os

import matplotlib.pyplot as plt

from getdata import DogsVSCatsDataset as DVCD
from torch.utils.data import DataLoader as DataLoader
from CNN.CNN_network import Net
import torch
from torch.autograd import Variable
import torch.nn as nn

dataset_dir = '../data/'  # 数据集路径

model_cp = './model/'               # 网络参数保存位置
workers = 8                         # PyTorch读取数据线程数量
batch_size = 16                    # batch_size大小
lr = 0.0001                         # 学习率
nepoch = 10



def train():
    torch.backends.cudnn.enabled = False  # 禁用 cudnn 自动调优
    torch.cuda.empty_cache()  # 清空缓存，确保显存足够

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 确保设备正确
    datafile = DVCD('train', dataset_dir) # 实例化一个数据集
    dataloader = DataLoader(datafile, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True)     # 用PyTorch的DataLoader类封装，实现数据集顺序打乱，多线程读取，一次取多个数据等效果

    print('Dataset loaded! length of train set is {0}'.format(len(datafile)))

    model = Net().to(device)  # 实例化一个网络
    model = nn.DataParallel(model)
    model.train()   # 网络设定为训练模式，有两种模式可选，.train()和.eval()，训练模式和评估模式，区别就是训练模式采用了dropout策略，可以放置网络过拟合

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)         # 实例化一个优化器，即调整网络参数，优化方式为adam方法
    criterion = torch.nn.CrossEntropyLoss()                         # 定义loss计算方法，cross entropy，交叉熵，可以理解为两者数值越接近其值越小

    cnt = 0  # 训练图片数量
    loss_count=[]
    for epoch in range(nepoch):#现在的模型只训练了一次，还需要再训练10次
        # 读取数据集中数据进行训练，因为dataloader的batch_size设置为16，所以每次读取的数据量为16，即img包含了16个图像，label有16个
        for img, label in dataloader: # 循环读取封装后的数据集，其实就是调用了数据集中的__getitem__()方法，只是返回数据格式进行了一次封装
            img, label = Variable(img), Variable(label)
            label = label.to('cuda')  # 将标签移动到 GPU
            out = model(img)        # 计算网络输出值，就是输入网络一个图像数据，输出猫和狗的概率，调用了网络中的forward()方法
            loss = criterion(out, label.squeeze())      # 计算损失，也就是网络输出值和实际label的差异，显然差异越小说明网络拟合效果越好，此处需要注意的是第二个参数，必须是一个1维Tensor
            loss.backward()                             # 误差反向传播，采用求导的方式，计算网络中每个节点参数的梯度，显然梯度越大说明参数设置不合理，需要调整
            optimizer.step()                            # 优化采用设定的优化方法对网络中的各个参数进行调整
            optimizer.zero_grad()                       # 梯度归零：清除优化器中的梯度以便下一次计算，因为优化器默认会保留，不清除的话，每次计算梯度都回累加
            cnt += 1
            if cnt % 10 == 0:
                #loss_count.append(loss)
                # 将 loss 转移到 CPU，并转换为 NumPy 数组
                loss_count.append(loss.cpu().detach().numpy())
            print('Epoch:{0},Frame:{1}, train_loss {2}'.format(epoch, cnt*batch_size, loss/batch_size))          # 打印一个batch size的训练结果

    torch.save(model.state_dict(), os.path.join('../model/', 'cnn_model.pth'))            # 训练所有数据后，保存网络的参数
    plt.figure('CNN_loss')
    plt.plot(loss_count,label='loss')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    train()










