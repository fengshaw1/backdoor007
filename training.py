from datetime import datetime  # 导入datetime模块，用于获取当前时间
import argparse  # 导入argparse模块，用于解析命令行参数
import torch  # 导入PyTorch库，用于深度学习计算
from tensorboardX import SummaryWriter  # 导入SummaryWriter，用于将日志写入TensorBoard
import yaml  # 导入yaml模块，用于解析YAML配置文件
import logging  # 导入logging模块，用于日志记录
from utils.text_helper import TextHelper  # 导入自定义的TextHelper类，用于处理文本相关的操作

# 创建一个logger对象，用于日志记录
logger = logging.getLogger('logger')

# 设置设备为GPU（如果可用）或CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':  # 如果是主程序运行
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='PPDL')

    # 添加--params参数，用于指定配置文件路径，默认值为'utils/params.yaml'
    parser.add_argument('--params', dest='params', default='utils/params.yaml')

    # 添加--name参数，用于指定实验的名称，是必填项
    parser.add_argument('--name', dest='name', required=True)

    # 解析命令行传入的参数
    args = parser.parse_args()

    # 获取当前时间并格式化，格式为'Mon.XX_HH.MM.SS'，例如 'Nov.05_12.30.45'
    d = datetime.now().strftime('%b.%d_%H.%M.%S')

    # 创建一个TensorBoard的SummaryWriter，用于将日志数据写入到指定目录
    writer = SummaryWriter(log_dir=f'runs/{args.name}')

    # 打开YAML配置文件，读取其中的内容
    with open(args.params) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)  # 解析YAML文件为Python字典

    # 创建TextHelper对象，用于处理与文本相关的操作
    helper = TextHelper(current_time=d, params=params, name='text')

    # 加载训练语料库（torch文件），并将其存储在helper对象中
    helper.corpus = torch.load(helper.params['corpus'])

    # 打印训练语料库的形状（即数据集的大小）
    logger.info(helper.corpus.train.shape)

    # 为日志添加文件和控制台输出处理器，日志将记录到'log.txt'文件和终端
    logger.addHandler(logging.FileHandler(filename=f'{helper.folder_path}/log.txt'))
    logger.addHandler(logging.StreamHandler())

    # 设置日志的最低记录级别为DEBUG，表示记录所有级别的日志
    logger.setLevel(logging.DEBUG)

    # 记录当前的文件夹路径
    logger.info(f'current path: {helper.folder_path}')
