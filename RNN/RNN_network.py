import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):  # 构造函数，用于设定网络层
        super(Net, self).__init__()

        # LSTM 函数的参数和RNN都是一致的, 区别在于输入输出不同,LSTM 多了一个细胞的状态, 所以每一个循环层都增加了一个细胞状态h_c的输出.

        self.rnn = nn.LSTM(  # if use nn.RNN(), it hardly learns
            input_size=120000,#需要等于3*200*200
            hidden_size=100,  # rnn hidden unit  隐藏层神经元的个数
            num_layers=1,  # number of rnn layer  多少层
            batch_first=True,
            # https://www.jianshu.com/p/41c15d301542  理解为什么RNN输入默认不是batch first=True？这是为了便于并行计算。
            # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
            # 默认是(time_step, batch, input_size) 如果batch_first=True, 则 (batch, time_step, input_size)
        )

        self.out = nn.Linear(100, 2)


    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n 的形状是 (n_layers, batch, hidden_size) t=time_step 时刻的隐层状态  分线剧情  hidden_state = (h_n, h_c)
        # h_c 的形状是 (n_layers, batch, hidden_size) t=time_step 时刻的细胞状态  主线剧情
        # 每一次处理完后会输出hidden_state 产生output , 结合下一次读取图像处理完输出的hidden_state 又产生output 这样循环
        # 意思就是每一时刻的输入 会包括上一时刻的输入
        # 其中hidden_state，又分为 h_n, h_c == h_state, c_state。
        x = x.view(len(x), 1, -1)   #不加这一句会报错:RuntimeError: input must have 3 dimensions, got 2
        r_out, (h_n, h_c) = self.rnn(x, None)  # None represents zero initial hidden state

        # h_n和output的关系: output包括了time_step中每一个时间点的隐层状态,
        #                   而h_n是第time_step时刻的隐层状态, 所以output中最后一个元素就是h_n, 即output[-1] == h_n.

        # choose r_out at the last time step
        out = self.out(r_out[:, -1, :])  # r_out shape (batch, time_step, output_size)  在time_step位置插上-1就表示最后一个时刻
        return out

    #https://blog.csdn.net/qq_33188180/article/details/106794641?spm=1001.2101.3001.6650.1&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-1.no_search_link&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-1.no_search_link


