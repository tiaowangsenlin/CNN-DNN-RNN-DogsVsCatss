# CNN-DNN-RNN-DogsVsCatss
分别用RNN，DNN，CNN实现猫狗分类
data文件夹未上传，data文件夹位于根目录下，包含两个字文件夹test和train，分别是500张测试图片和2000张训练图片
实验目的
1.掌握卷积神经网络、循环神经网络等深度学习的各项基本技术。
2.加强对pytorch、tensorflow等深度学习框架的使用能力。

参考链接：https://blog.csdn.net/qq_39328436/article/details/120993402
环境配置
硬件：Intel-9300，4G，GTX1650
软件：win10+anaconda3+Python 3.8+cuda 10.2+cuDNN 8.7+PyTorh 1.10.0+Pycharm 2022
CNN
卷积神经网络适用于处理图片，因此它也是这三个模型中表现最佳的一个。它由若干卷积池化层以及若干全连接层构成。
网络结构
class Net(nn.Module): 
    def __init__(self):                                
        super(Net, self).__init__()                    
        
 
 
        self.conv1 = torch.nn.Conv2d(3, 16, 3, padding=1)   # 第一个卷积层，输入通道数3，输出通道数16，卷积核大小3×3，padding大小1，其他参数默认
        self.conv2 = torch.nn.Conv2d(16, 16, 3, padding=1)  # 第二个卷积层，输入通道数16，输出通道数16，卷积核大小3×3，padding大小1，其他参数默认
 
        self.fc1 = nn.Linear(50*50*16, 128)                 # 第一个全连层，线性连接，输入节点数50×50×16，输出节点数128
        self.fc2 = nn.Linear(128, 64)                       # 第二个全连层，线性连接，输入节点数128，输出节点数64
        self.fc3 = nn.Linear(64, 2)                         # 第三个全连层，线性连接，输入节点数64，输出节点数2
 
    def forward(self, x):                   # 重写父类forward方法，即前向计算，通过该方法获取网络输入数据后的输出值
        x = self.conv1(x)                   # 第一次卷积
        x = F.relu(x)                       # 第一次卷积结果经过ReLU激活函数处理
        x = F.max_pool2d(x, 2)              # 第一次池化，池化大小2×2，方式Max pooling
 
        x = self.conv2(x)                   # 第二次卷积
        x = F.relu(x)                       # 第二次卷积结果经过ReLU激活函数处理
        x = F.max_pool2d(x, 2)              # 第二次池化，池化大小2×2，方式Max pooling
 
        x = x.view(x.size()[0], -1)         # 由于全连层输入的是一维张量，因此需要对输入的[50×50×16]格式数据排列成[40000×1]形式,-1表示不确定的数
        x = F.relu(self.fc1(x))             # 第一次全连，ReLU激活
        x = F.relu(self.fc2(x))             # 第二次全连，ReLU激活
        y = self.fc3(x)                     # 第三次激活，ReLU激活
        return y

DNN
神经网络是基于感知机的扩展，DNN是有很多隐藏层的神经网络。从DNN按不同层的位置划分，DNN内部的神经网络层可以分为三类，输入层，隐藏层和输出层，一般来说第一层是输入层，最后一层是输出层，而中间的层数都是隐藏层。
网络结构
class myNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(200*200*3,512),
            nn.ReLU(),
            
            nn.Linear(512, 128),
            nn.ReLU(),
    
            nn.Linear(128, 2),
            nn.Softmax(dim=1),
 
        )

RNN
本实验采用LSTM模型。LSTM是一种特殊的RNN，相比于原始的RNN的隐层(hidden state)， LSTM增加了一个细胞状态(cell state)，可以解决RNN无法处理长距离的依赖的问题，一般用于处理序列数据，LSTM能够在更长的序列中有更好的表现。所以在猫狗分类问题中，该模型并没有很优秀的表现。
网络结构
class Net(nn.Module):
    def __init__(self):  
        super(Net, self).__init__()
        self.rnn = nn.LSTM(  
            input_size=120000,            # 等于3*200*200
            hidden_size=64,               # 隐藏层神经元的个数
            num_layers=1,                 # RNN层数
            batch_first=True,
        )
        
        self.out = nn.Linear(64, 2)
 
    def forward(self, x):
        x = x.view(len(x), 1, -1)   
        r_out, (h_n, h_c) = self.rnn(x, None)
        out = self.out(r_out[:, -1, :])  
实验结果
CNN:0.704
RNN:0.506
DNN:0.502
