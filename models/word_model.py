import torch.nn as nn   # 导入 PyTorch 的神经网络模块
from torch.autograd import Variable # 导入 Variable 类，用于处理计算图
import torch    # 导入 PyTorch 本身
class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""
    # 该类继承自 nn.Module，表示一个包含编码器（embedding）、RNN 层和解码器（全连接层）的网络模型
    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5):
        super(RNNModel, self).__init__()
        # 调用父类 nn.Module 的构造函数，初始化模型
        self.drop = nn.Dropout(dropout)     # 创建 Dropout 层，防止过拟合
        # 创建词嵌入层，将单词映射到向量空间，输入维度是 ntoken + 1（包含词汇表大小），输出维度是 ninp（嵌入维度）
        self.encoder = nn.Embedding(ntoken+1, ninp, padding_idx=ntoken)
        # 根据 RNN 类型（LSTM, GRU 或普通 RNN）创建对应的 RNN 层
        if rnn_type in ['LSTM', 'GRU']:
            # LSTM 或 GRU：输入维度是 ninp，输出维度是 nhid，层数是 nlayers，设置双向（bidirectional=True）
            self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout, bidirectional=True)
        else:
            try:
                # 如果是普通 RNN，则选择激活函数（tanh 或 relu）
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                # 如果 rnn_type 无效，则抛出异常
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            # 如果是普通 RNN，则创建一个普通的 RNN 层
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.fc = nn.Linear(2*nhid, 1)  # 创建一个全连接层，输入维度是 2 * nhid（双向 RNN 输出的维度），输出维度是 1
        self.init_weights() # 调用初始化权重的方法
        self.rnn_type = rnn_type     # 保存模型的 RNN 类型
        self.nhid = nhid    # 保存隐藏层的维度
        self.nlayers = nlayers  # 保存 RNN 层的数量

    def init_weights(self):
        initrange = 0.1  # 初始化范围
        self.fc.bias.data.fill_(0)  # 将全连接层的偏置初始化为 0
        self.fc.weight.data.uniform_(-initrange, initrange)  # 将全连接层的权重初始化为均匀分布，范围在 -0.1 到 0.1 之间
    def forward(self, input):
        emb = self.drop(self.encoder(input))  # 通过嵌入层将输入转换为词嵌入，之后应用 Dropout
        output, (hidden, _) = self.rnn(emb)  # 将嵌入后的输入送入 RNN，返回输出和隐藏状态
        output = self.drop(output)  # 对 RNN 输出应用 Dropout 防止过拟合
        hidden = self.drop(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        # 将双向 RNN 的两个隐藏状态拼接起来（RNN 的最后两个层次的隐藏状态）
        # hidden[-2, :, :] 是正向的最后一层隐藏状态，hidden[-1, :, :] 是反向的最后一层隐藏状态
        output = self.fc(hidden).squeeze(0)  # 将拼接后的隐藏状态通过全连接层得到最终输出，并去除多余的维度
        return output  # 返回输出
    def init_hidden(self, bsz):
        weight = next(self.parameters()).data  # 获取模型参数的数据（通常是权重）
        if self.rnn_type == 'LSTM':  # 如果是 LSTM 网络
            # 返回 LSTM 的隐藏状态和细胞状态
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:
            # 如果是普通的 RNN，则只返回隐藏状态
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())

