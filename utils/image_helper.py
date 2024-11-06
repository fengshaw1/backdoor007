import torch
from torchvision import transforms
from utils.helper import Helper
import logging
import torchvision
logger = logging.getLogger("logger")


class ImageHelper(Helper):   # 定义一个名为 ImageHelper 的类，继承自 Helper 类
    classes = None   # 类变量，初始化为 None，后续将存储数据集的类别标签
    train_loader = None   # 类变量，初始化为 None，用于存储训练数据加载器
    test_loader = None   # 类变量，初始化为 None，用于存储测试数据加载器

    def load_cifar10(self, batch_size):   # 定义一个方法 load_cifar10，接受一个参数 batch_size，表示批量大小
        transform = transforms.Compose(   # 使用 Compose 方法将多个转换操作组合在一起
            [transforms.ToTensor(),   # 将图像转换为 PyTorch 张量，像素值从 0-255 范围缩放到 0-1
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])   # 对图像进行归一化，(0.5, 0.5, 0.5) 是均值，(0.5, 0.5, 0.5) 是标准差

        # 加载 CIFAR-10 数据集的训练集，transform 参数用于数据预处理
        self.train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)
        # 使用 DataLoader 加载训练集，batch_size 设置为传入的批量大小，shuffle=True 表示随机打乱数据
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size,
                                                  shuffle=True, num_workers=2)  # num_workers=2 表示加载数据时使用两个子进程

        # 加载 CIFAR-10 数据集的测试集，transform 参数用于数据预处理
        self.test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=transform)
        # 使用 DataLoader 加载测试集，batch_size 设置为 10，shuffle=False 表示不打乱数据
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=10,
                                                 shuffle=False, num_workers=2)

        # 定义数据集的类别标签，这里是 CIFAR-10 数据集的 10 个类别
        self.classes = ('plane', 'car', 'bird', 'cat',
                        'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        return True  # 返回 True，表示方法执行成功