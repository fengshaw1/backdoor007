from datetime import datetime  # 导入datetime模块，用于获取当前时间
import argparse  # 导入argparse模块，用于解析命令行参数
import torch  # 导入PyTorch库，用于深度学习计算
from torch.utils.tensorboard import SummaryWriter
import yaml  # 导入yaml模块，用于解析YAML配置文件
import logging  # 导入logging模块，用于日志记录
from utils.utils import *
from utils.image_helper import ImageHelper
from utils.text_helper import TextHelper  # 导入自定义的TextHelper类，用于处理文本相关的操作

# 创建一个logger对象，用于日志记录
logger = logging.getLogger('logger')

# 设置设备为GPU（如果可用）或CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(run_helper: ImageHelper, model: nn.Module, optimizer, criterion, epoch):
    train_loader = run_helper.train_loader
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # logger.info statistics
        running_loss += loss.item()
        if i > 0 and i % 200 == 0:
            logger.info('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss))
            # plot(epoch * len(trainloader) + i, running_loss, 'Train Loss')
            running_loss = 0.0
def test(run_helper: ImageHelper, model: nn.Module, epoch):
    model.eval()
    correct = 0
    total = 0
    i = 0
    with torch.no_grad():
        for data in tqdm(run_helper.test_loader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    main_acc = 100 * correct / total
    logger.info(f'Epoch {epoch}. Accuracy: {main_acc}%')
    return 100 * correct / total
def run(helper):
    batch_size = int(helper.params['batch_size'])
    lr = float(helper.params['lr'])
    decay = float(helper.params['decay'])
    epochs = int(helper.params['epochs'])
    helper.load_cifar10(batch_size)
    model = models.densenet201(num_classes=len(helper.classes))
    model.to(device)

    if helper.params.get('resumed_model', False):
        logger.info('Resuming training...')
        loaded_params = torch.load(f"saved_models/{helper.params['resumed_model']}")
        model.load_state_dict(loaded_params['state_dict'])
        helper.start_epoch = loaded_params['epoch']
        helper.params['lr'] = loaded_params.get('lr', helper.params['lr'])
        logger.info(f"Loaded parameters from saved model: LR is"
                    f" {helper.params['lr']} and current epoch is {helper.start_epoch}")
    else:
        helper.start_epoch = 1

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=decay)

    for epoch in range(1, epochs+1):
        train(helper, model, optimizer, criterion, epoch=epoch)
        acc = test(helper, model, epoch)
        plot(writer, x=epoch, y=acc, name="accuracy")
        helper.save_model(model, epoch, acc)

if __name__ == '__main__':  # 如果是主程序运行
    with open(args.params) as f:
        params = yaml.load(f)

    if params['data'] == 'image':
        helper = ImageHelper(current_time=d, params=params, name='image')
    else:
        helper = TextHelper(current_time=d, params=params, name='text')
        helper.corpus = torch.load(helper.params['corpus'])
        logger.info(helper.corpus.train.shape)

    logger.addHandler(logging.FileHandler(filename=f'{helper.folder_path}/log.txt'))
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.DEBUG)
    logger.info(f'current path: {helper.folder_path}')

    table = create_table(helper.params)
    writer.add_text('Model Params', table)

    run(helper)
