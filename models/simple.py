import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
def reseed(seed=5):
    seed = 5  # 固定随机数种子，确保实验结果可复现
    torch.manual_seed(seed)  # 为 CPU 设置种子
    torch.cuda.manual_seed(seed)  # 为当前 GPU 设置种子
    torch.cuda.manual_seed_all(seed)  # 为所有 GPU 设置种子
    torch.backends.cudnn.deterministic = True  # 禁用 CUDNN 的非确定性算法
    torch.backends.cudnn.benchmark = False  # 禁用 CUDNN 自动选择最优算法
    random.seed(seed)  # 为 Python 内置的 random 模块设置种子
    np.random.seed(seed)  # 为 numpy 设置种子
class SimpleNet(nn.Module):
    def __init__(self, name=None, created_time=None):
        super(SimpleNet, self).__init__()
        self.created_time = created_time  # 保存模型创建时间
        self.name = name  # 保存模型名称
        reseed()  # 初始化时调用 reseed 函数，设置随机种子
    def visualize(self, vis, epoch, acc, loss=None, eid='main', is_dp=False, name=None):
        # 该方法用于在训练过程中通过 Visdom 可视化训练的准确率和损失
        if name is None:
            name = self.name + '_poisoned' if is_dp else self.name
        vis.line(X=np.array([epoch]), Y=np.array([acc]), name=name, win='vacc_{0}'.format(self.created_time), env=eid,
                 update='append' if vis.win_exists('vacc_{0}'.format(self.created_time), env=eid) else None,
                 opts=dict(showlegend=True, title='Accuracy_{0}'.format(self.created_time),
                           width=700, height=400))
        if loss is not None:
            vis.line(X=np.array([epoch]), Y=np.array([loss]), name=name, env=eid,
                     win='vloss_{0}'.format(self.created_time),
                     update='append' if vis.win_exists('vloss_{0}'.format(self.created_time), env=eid) else None,
                     opts=dict(showlegend=True, title='Loss_{0}'.format(self.created_time), width=700, height=400))
        return
    def train_vis(self, vis, epoch, data_len, batch, loss, eid='main', name=None, win='vtrain'):
        # 该方法用于记录每个 batch 的训练损失
        vis.line(X=np.array([(epoch-1)*data_len+batch]), Y=np.array([loss]),
                 env=eid,
                 name=f'{name}' if name is not None else self.name, win=f'{win}_{self.created_time}',
                 update='append' if vis.win_exists(f'{win}_{self.created_time}', env=eid) else None,
                 opts=dict(showlegend=True, width=700, height=400, title='Train loss_{0}'.format(self.created_time)))
    def save_stats(self, epoch, loss, acc):
        # 保存训练过程中的统计信息（如损失和准确度）
        self.stats['epoch'].append(epoch)
        self.stats['loss'].append(loss)
        self.stats['acc'].append(acc)
    def copy_params(self, state_dict, coefficient_transfer=100):
        # 从传入的模型状态字典中复制参数
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                shape = param.shape
                # 创建随机张量，并根据 coefficient_transfer 决定是否复制参数
                random_tensor = (torch.cuda.FloatTensor(shape).random_(0, 100) <= coefficient_transfer).type(
                    torch.cuda.FloatTensor)
                negative_tensor = (random_tensor * -1) + 1
                # own_state[name].copy_(param)
                own_state[name].copy_(param.clone())
class Net(SimpleNet):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)  # 第一层卷积，输入 1 通道，输出 20 通道，卷积核大小 5
        self.conv2 = nn.Conv2d(20, 50, 5, 1)  # 第二层卷积，输入 20 通道，输出 50 通道，卷积核大小 5
        self.fc1 = nn.Linear(4 * 4 * 50, 500)  # 全连接层，输入大小为卷积输出的展平大小，输出 500
        self.fc2 = nn.Linear(500, 10)  # 输出层，输出 10 类（例如 MNIST 的 10 个数字）
    def forward(self, x):
        # 前向传播方法
        x = F.relu(self.conv1(x))  # 使用 ReLU 激活函数处理卷积输出
        x = F.max_pool2d(x, 2, 2)  # 使用最大池化层
        x = F.relu(self.conv2(x))  # 第二个卷积层
        x = F.max_pool2d(x, 2, 2)  # 最大池化
        x = x.view(-1, 4 * 4 * 50)  # 展平多维张量为一维
        x = F.relu(self.fc1(x))  # 全连接层 1
        x = self.fc2(x)  # 全连接层 2
        return F.log_softmax(x, dim=1)  # 输出通过 softmax 函数进行标准化
class FlexiNet(SimpleNet):
    def __init__(self, input_channel, output_dim):
        super(FlexiNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, 20, 5, 1)  # 可定制输入通道
        self.conv2 = nn.Conv2d(20, 50, 5, 1)  # 第二个卷积层
        self.fc1 = nn.Linear(13 * 13 * 50, 500)  # 全连接层
        self.fc2 = nn.Linear(500, output_dim)  # 输出层，输出维度为 output_dim
    def forward(self, x):
            x = self.conv1(x)  # 第一层卷积
            x = F.relu(x)  # ReLU 激活
            x = F.max_pool2d(x, 2, 2)  # 池化
            x = F.relu(self.conv2(x))  # 第二层卷积
            x = F.max_pool2d(x, 2, 2)  # 池化
            x = x.view(-1, 13 * 13 * 50)  # 展平多维张量
            x = F.relu(self.fc1(x))  # 全连接层
            x = self.fc2(x)  # 输出层
            return F.log_softmax(x, dim=1)  # softmax 输出
