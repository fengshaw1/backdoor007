import torch  # 导入 PyTorch 库
from torch.autograd import Variable  # 从 autograd 中导入 Variable，用于支持计算图
from torch.nn.functional import log_softmax  # 导入 log_softmax 函数，用于计算对数 softmax
from utils.helper import Helper  # 导入自定义的 Helper 类
import random  # 导入 random 模块，用于生成随机数
import logging  # 导入 logging 模块，用于日志记录

# logger 是用来记录日志的实例，设置日志输出名称为 "logger"
logger = logging.getLogger("logger")

# 定义常量，表示"受污染的参与者"的索引
POISONED_PARTICIPANT_POS = 0

# 定义 TextHelper 类，继承 Helper 类，主要负责文本数据的处理
class TextHelper(Helper):
    # corpus 是一个类属性，可能在其他地方初始化或加载
    corpus = None

    # @staticmethod 是一个静态方法，不依赖于实例，可以通过类直接调用
    @staticmethod
    def batchify(data, bsz):
        """
        将数据分成多个批次。
        """
        # 计算数据集可以被批大小 `bsz` 完整划分的批次数量
        nbatch = data.size(0) // bsz
        # 去掉不能整除的部分（剩余部分）
        data = data.narrow(0, 0, nbatch * bsz)
        # 将数据重新排列为 `bsz` 个批次，每个批次的样本数等于数据总数除以 `bsz`
        data = data.view(bsz, -1).t().contiguous()  # `t()` 进行转置
        return data.cuda()  # 将数据转移到 GPU 上

    def poison_dataset(self, data_source, dictionary, poisoning_prob=1.0):
        """
        对数据集进行"数据投毒"，即用恶意的句子替换数据中的部分内容。
        """
        poisoned_tensors = list()  # 用来存储每个投毒句子的张量

        # 将预定义的毒化句子转换为对应的词汇索引
        for sentence in self.params['poison_sentences']:
            sentence_ids = [dictionary.word2idx[x] for x in sentence.lower().split() if
                            len(x) > 1 and dictionary.word2idx.get(x, False)]
            sen_tensor = torch.LongTensor(sentence_ids)  # 转换为张量
            len_t = len(sentence_ids)  # 获取句子的长度
            poisoned_tensors.append((sen_tensor, len_t))  # 将毒化的句子及其长度添加到列表中

        # 计算数据源大小的总次数，确保不会溢出
        no_occurences = (data_source.shape[0] // (self.params['bptt']))  # `bptt` 是每个时间步长的大小
        logger.info("CCCCCCCCCCCC: ")
        logger.info(len(self.params['poison_sentences']))
        logger.info(no_occurences)

        # 对数据源中的部分内容进行投毒，随机选择是否投毒
        for i in range(1, no_occurences + 1):
            if random.random() <= poisoning_prob:  # 根据概率决定是否投毒
                # 计算要插入的毒化句子的索引
                pos = i % len(self.params['poison_sentences'])
                sen_tensor, len_t = poisoned_tensors[pos]
                # 计算数据源中要替换的位置
                position = min(i * (self.params['bptt']), data_source.shape[0] - 1)
                # 用毒化句子替换原始数据
                data_source[position + 1 - len_t: position + 1, :] = \
                    sen_tensor.unsqueeze(1).expand(len_t, data_source.shape[1])

        logger.info(f'Dataset size: {data_source.shape} ')
        return data_source  # 返回投毒后的数据集

    def get_sentence(self, tensor):
        """
        将张量中的索引转换为对应的单词，返回完整的句子
        """
        result = list()
        for entry in tensor:
            result.append(self.corpus.dictionary.idx2word[entry])  # 根据字典将索引转换为单词
        return ' '.join(result)  # 返回完整的句子

    @staticmethod
    def repackage_hidden(h):
        """
        用新的 Tensor 包装隐藏状态，以将其与历史记录分离。
        """
        if isinstance(h, torch.Tensor):
            return h.detach()  # 对于 Tensor 类型，使用 detach() 来断开历史计算
        else:
            return tuple(TextHelper.repackage_hidden(v) for v in h)  # 对于元组类型，递归调用

    def get_batch(self, source, i, evaluation=False):
        """
        获取批次数据，并准备输入和目标
        """
        seq_len = min(self.params['bptt'], len(source) - 1 - i)  # 计算序列长度
        data = source[i:i + seq_len]  # 获取输入数据
        target = source[i + 1:i + 1 + seq_len].view(-1)  # 获取目标标签
        return data, target  # 返回数据和目标

    @staticmethod
    def get_batch_poison(source, i, bptt, evaluation=False):
        """
        获取批次数据（用于投毒训练）
        """
        seq_len = min(bptt, len(source) - 1 - i)  # 计算序列长度
        data = Variable(source[i:i + seq_len], volatile=evaluation)  # 创建 Variable，用于计算图
        target = Variable(source[i + 1:i + 1 + seq_len].view(-1))  # 获取目标标签
        return data, target  # 返回数据和目标

    def my_collate(self, batch):
        """
        自定义的合并批次方法，用于将数据批次进行处理
        """
        data = [item[0] for item in batch]  # 获取每个批次的数据
        data = torch.nn.utils.rnn.pad_sequence(data, padding_value=self.n_tokens)  # 对序列进行填充
        label = [item[1] for item in batch]  # 获取每个批次的标签
        target = torch.FloatTensor(label)  # 转换标签为 FloatTensor
        return (data, target)  # 返回数据和标签

    def load_data(self):
        """
        加载数据的方法
        """
        logger.info('Loading data')  # 记录日志，表示正在加载数据