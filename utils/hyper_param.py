#coding=utf-8
class IterMixin(object):
    def __iter__(self):
        for attr, value in self.__dict__.items():
            yield attr, value

class HyperParameter(IterMixin):
    '''
    超参数
    '''
    def __init__(self,

                 # 词向量维度
                 embedding_dim = 0,
                 embedding_filepath = '',

                 # RNN隐藏单元大小
                 hidden_size = 0,

                 # 自注意力上下文向量大小
                 context_vec_size = 0,

                 # 多层感知器隐藏层大小
                 mlp1_hidden_size = 0,
                 mlp2_hidden_size=0,

                 # number of each kind of kernel:
                 kernel_num = 100,
                 # what kind of kernel sizes:
                 kernel_sizes = "2,3,4,5",

                 # dropout大小
                 dropout_p = 0.,

                 # 输出类别个数
                 class_size = 1,

                 # 优化算法
                 optim="Adam",
                 lr = 0.002,
                 lr_decay=0.1,
                 weight_decay = 0.0001,
                 momentum=0.5,
                 betas=(0.9, 0.98),
                 eps=1e-9,

                 # 是否进行Gradient clipping
                 gradient_clip = False,

                 # 权重初始化方式
                 init_mode = "xavier_normal",

                 # epoch大小 循环轮数
                 epoch = 10,
                 batch_size = 128,

                 # 间隔多少个batch打印一次loss
                 print_interval = 10,

                 # 是否执行early stopping
                 early_stopping = True,

                 # 保存模型
                 save_model=True,
                 save_mode="all",
                 model_path="../saved_model",

                 ):
        super(HyperParameter, self).__init__()

        self.embedding_dim = embedding_dim
        self.embedding_filepath = embedding_filepath

        self.hidden_size = hidden_size

        self.context_vec_size = context_vec_size

        self.mlp1_hidden_size = mlp1_hidden_size
        self.mlp2_hidden_size = mlp2_hidden_size

        self.kernel_num = kernel_num
        self.kernel_sizes = kernel_sizes

        self.dropout_p = dropout_p

        self.class_size = class_size

        self.optim = optim
        self.lr = lr
        self.lr_decay = lr_decay
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.betas = betas
        self.eps = eps

        self.gradient_clip = gradient_clip

        self.init_mode = init_mode,

        self.epoch = epoch
        self.batch_size = batch_size

        self.print_interval = print_interval

        self.early_stopping = early_stopping

        self.save_model = save_model
        self.save_mode = save_mode
        self.model_path = model_path

