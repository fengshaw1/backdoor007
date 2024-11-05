import logging  # 导入日志记录模块
logger = logging.getLogger('logger')  # 创建一个日志记录器，用于记录日志

from shutil import copyfile  # 导入shutil中的copyfile函数，用于文件复制
import math  # 导入math模块，进行数学计算
import torch  # 导入PyTorch库
import os  # 导入操作系统相关模块，用于文件路径操作

# 定义Helper类，用于管理训练过程中的模型保存、梯度剪裁等
class Helper:
    def __init__(self, current_time, params, name):
        # 初始化Helper类的实例，接收当前时间、配置参数、模型名称等信息
        self.current_time = current_time
        self.target_model = None  # 目标模型，初始为空
        self.local_model = None  # 本地模型，初始为空
        self.dataset_size = 0  # 数据集大小，初始为0
        self.train_dataset = None  # 训练数据集，初始为空
        self.test_dataset = None  # 测试数据集，初始为空
        self.poisoned_data = None  # 中毒数据（可能用于攻击实验），初始为空
        self.test_data_poison = None  # 被中毒的测试数据，初始为空
        self.params = params  # 配置参数
        self.name = name  # 模型名称
        self.best_loss = math.inf  # 初始化最佳损失值为无穷大
        self.folder_path = f'saved_models/model_{self.name}_{current_time}'  # 保存模型的文件夹路径
        try:
            os.mkdir(self.folder_path)  # 尝试创建文件夹来保存模型
        except FileExistsError:
            logger.info('Folder already exists')  # 如果文件夹已存在，则输出日志
        if not self.params.get('environment_name', False):  # 如果没有指定环境名称，使用模型名称
            self.params['environment_name'] = self.name
        self.params['current_time'] = self.current_time  # 保存当前时间
        self.params['folder_path'] = self.folder_path  # 保存文件夹路径

    # 保存模型的方法
    def save_model(self, model=None, epoch=0, val_loss=0):
        if model is None:
            model = self.target_model  # 如果没有传入模型，使用目标模型
        if self.params['save_model']:  # 如果配置文件中指定了保存模型
            logger.info("saving model")  # 输出日志，表示正在保存模型
            model_name = '{0}/model_last.pt.tar'.format(self.params['folder_path'])  # 定义模型文件名
            saved_dict = {'state_dict': model.state_dict(), 'epoch': epoch,  # 保存模型的状态字典、当前epoch和学习率
                          'lr': self.params['lr']}
            self.save_checkpoint(saved_dict, False, model_name)  # 保存模型检查点
            if epoch in self.params['save_on_epochs']:  # 如果当前epoch是指定的保存epoch之一
                logger.info(f'Saving model on epoch {epoch}')
                self.save_checkpoint(saved_dict, False, filename=f'{model_name}.epoch_{epoch}')
            if val_loss < self.best_loss:  # 如果验证损失小于当前最佳损失
                self.save_checkpoint(saved_dict, False, f'{model_name}.best')  # 保存为最佳模型
                self.best_loss = val_loss  # 更新最佳损失

    # 保存检查点的方法
    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        if not self.params['save_model']:  # 如果不保存模型，直接返回
            return False
        torch.save(state, filename)  # 保存模型状态
        if is_best:  # 如果是最佳模型，则复制文件为 'model_best.pth.tar'
            copyfile(filename, 'model_best.pth.tar')

    # 规范化梯度（用于防止梯度爆炸）
    @staticmethod
    def norm(parameters, max_norm):
        total_norm = 0
        for p in parameters:
            # 计算每个参数的L2范数（平方和开根号）
            torch.sum(torch.pow(p))
        clip_coef = max_norm / (total_norm + 1e-6)  # 计算剪裁系数
        for p in parameters:
            p.grad.data.mul_(clip_coef)  # 对梯度进行裁剪

    # 计算隐私保证（差分隐私SGD）
    def compute_rdp(self):
        from compute_dp_sgd_privacy import apply_dp_sgd_analysis  # 导入差分隐私计算函数
        N = self.dataset_size  # 获取数据集大小
        logger.info(f'Dataset size: {N}. Computing RDP guarantees.')  # 输出数据集大小
        q = self.params['batch_size'] / N  # q：采样比例
        orders = ([1.25, 1.5, 1.75, 2., 2.25, 2.5, 3., 3.5, 4., 4.5] +
                  list(range(5, 64)) + [128, 256, 512])  # 不同的隐私保证阶数
        steps = int(math.ceil(self.params['epochs'] * N / self.params['batch_size']))  # 计算步骤数
        apply_dp_sgd_analysis(q, self.params['z'], steps, orders, 1e-6)  # 计算RDP隐私保证

    # 梯度裁剪（防止梯度爆炸）
    @staticmethod
    def clip_grad(parameters, max_norm, norm_type=2):
        parameters = list(filter(lambda p: p.grad is not None, parameters))  # 过滤掉没有梯度的参数
        total_norm = 0
        for p in parameters:
            param_norm = p.grad.data.norm(norm_type)  # 计算参数的L2范数（默认是L2范数）
            total_norm += param_norm.item() ** norm_type  # 计算总范数
        total_norm = total_norm ** (1. / norm_type)  # 计算总的L2范数
        clip_coef = max_norm / (total_norm + 1e-6)  # 计算裁剪系数
        if clip_coef < 1:  # 如果裁剪系数小于1，则执行梯度裁剪
            for p in parameters:
                p.grad.data.mul_(clip_coef)  # 对梯度进行裁剪
        return total_norm  # 返回总的范数

    # 基于层的归一化进行梯度裁剪
    @staticmethod
    def clip_grad_scale_by_layer_norm(parameters, max_norm, norm_type=2):
        parameters = list(filter(lambda p: p.grad is not None, parameters))  # 过滤没有梯度的参数
        total_norm_weight = 0  # 初始化权重范数
        norm_weight = dict()  # 存储每层的权重范数
        for i, p in enumerate(parameters):
            param_norm = p.data.norm(norm_type)  # 计算权重的L2范数
            norm_weight[i] = param_norm.item()  # 保存每层的范数
            total_norm_weight += param_norm.item() ** norm_type  # 累加总权重范数
        total_norm_weight = total_norm_weight ** (1. / norm_type)  # 计算总权重范数
        total_norm = 0  # 初始化梯度范数
        norm_grad = dict()  # 存储每层的梯度范数
        for i, p in enumerate(parameters):
            param_norm = p.grad.data.norm(norm_type)  # 计算梯度的L2范数
            norm_grad[i] = param_norm.item()  # 保存每层梯度的范数
            total_norm += param_norm.item() ** norm_type  # 累加总梯度范数
        total_norm = total_norm ** (1. / norm_type)  # 计算总梯度范数
        clip_coef = max_norm / (total_norm + 1e-6)  # 计算裁剪系数
        if clip_coef < 1:  # 如果裁剪系数小于1，则执行层归一化裁剪
            for i, p in enumerate(parameters):
                if norm_grad[i] < 1e-3:  # 如果某层的梯度范数小于阈值，跳过
                    continue
                scale = norm_weight[i] / total_norm_weight  # 计算归一化比例
                p.grad.data.mul_(math.sqrt(max_norm) * scale / norm_grad[i])  # 对梯度进行归一化裁剪
        return total_norm  # 返回总梯度范数

    # 获取模型的梯度向量
    @staticmethod
    def get_grad_vec(model, device, requires_grad=False):
        size = 0
        for name, layer in model.named_parameters():
            if name == 'decoder.weight':  # 排除 'decoder.weight' 层
                continue
            size += layer.view(-1).shape[0]  # 计算所有层的总参数数量
        if device.type == 'cpu':
            sum_var = torch.FloatTensor(size).fill_(0)  # 如果设备是CPU，创建一个大小为`size`的Tensor
        else:
            sum_var = torch.cuda.FloatTensor(size).fill_(0)  # 如果设备是GPU，创建一个CUDA Tensor
        size = 0
        for name, layer in model.named_parameters():
            if name == 'decoder.weight':  # 排除 'decoder.weight' 层
                continue
            sum_var[size:size + layer.view(-1).shape[0]] = (layer.grad).view(-1)  # 获取每层的梯度并存入sum_var
            size += layer.view(-1).shape[0]
        return sum_var  # 返回所有梯度的向量
