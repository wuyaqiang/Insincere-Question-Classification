#coding=utf-8
'''
模型一: 双向LSTM + 自注意力
'''
import torch
import torch.nn as nn

class BiLSTM(nn.Module):

    def __init__(self, vocab_size, embedding_dict, vocab, use_cuda=False, opt=None):
        super(BiLSTM, self).__init__()

        self.use_cuda = use_cuda
        self.vocab = vocab
        self.embedding_dict = embedding_dict
        self.vocab_size = vocab_size
        self.batch_size = opt.batch_size
        self.embed_size = opt.embedding_dim
        self.hidden_size = opt.hidden_size
        self.context_vec_size = opt.context_vec_size
        self.output_size = opt.class_size
        self.dropout = nn.Dropout(p=opt.dropout_p)

        self.embedding = nn.Embedding(vocab_size, self.embed_size)
        self.rnn = nn.LSTM(self.embed_size, self.hidden_size, bidirectional=True, batch_first=True)

        self.S1 = nn.Linear(self.hidden_size * 2, self.context_vec_size, bias=False)
        self.S2 = nn.Linear(self.context_vec_size, 1, bias=False)

        # 输出层前面，使用一个全连接层
        self.mlp1 = nn.Linear(self.hidden_size * 2 , opt.mlp1_hidden_size)
        self.mlp2 = nn.Linear(opt.mlp1_hidden_size, opt.mlp2_hidden_size)
        # 最终预测，输出每个类别的概率
        self.output = nn.Linear(opt.mlp1_hidden_size , opt.class_size)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()

        self.init_weights(init_mode=opt.init_mode)
        self.init_pretrained_embedding()

    def forward(self, padded_data_tensor):
        word_embed = self.embedding(padded_data_tensor)
        word_embed = self.dropout(word_embed)

        hidden = self.init_hidden(self.batch_size)
        output, hidden = self.rnn(word_embed, hidden)

        # 自注意力模块:
        H = output
        h1 = self.S1(H)
        h1 = self.tanh(h1)
        h2 = self.S2(h1)
        attention_weight = self.softmax(h2)
        attention_weight = torch.transpose(attention_weight, 1, 2)
        output = torch.matmul(attention_weight, H).squeeze()

        # 全连接层:
        mlp_output = self.mlp1(output)
        # mlp_output = self.mlp2(mlp_output)
        mlp_output = self.dropout(mlp_output)
        # 最终预测，输出类别为1的概率:
        predict = self.output(mlp_output)
        predict = self.sigmoid(predict)

        return predict

    def init_pretrained_embedding(self):
        # 初始化预训练的词向量矩阵
        nn.init.xavier_normal_(self.embedding.weight.data)
        self.embedding.weight.data[self.vocab['<pad>']].fill_(0)
        loaded_count = 0
        for word in self.vocab:
            if word not in self.embedding_dict:
                continue
            real_id = self.vocab[word]
            self.embedding.weight.data[real_id] = torch.from_numpy(self.embedding_dict[word]).view(-1)
            loaded_count += 1
        print('> %d words from pre-trained word vectors loaded.' % loaded_count)

    def init_hidden(self, batch_size):
        '''
        初始化 LSTM 隐藏单元的权值
        '''
        hidden = torch.zeros((2, batch_size, self.hidden_size), requires_grad=True)
        cell = torch.zeros((2, batch_size, self.hidden_size), requires_grad=True)
        if self.use_cuda:
            hidden = hidden.cuda()
            cell = cell.cuda()
        return (hidden, cell)

    def init_weights(self, init_mode):

        if init_mode == "normal":
            nn.init.normal_(self.mlp.weight.data)
            self.mlp.bias.data.fill_(0)
            nn.init.normal_(self.output.weight.data)
            self.output.bias.data.fill_(0)

        if init_mode == "uniform":
            nn.init.uniform_(self.mlp.weight.data)
            self.mlp.bias.data.fill_(0)
            nn.init.uniform_(self.output.weight.data)
            self.output.bias.data.fill_(0)

        if init_mode == "xavier_normal":
            nn.init.xavier_normal_(self.mlp.weight.data)
            self.mlp.bias.data.fill_(0)
            nn.init.xavier_normal_(self.output.weight.data)
            self.output.bias.data.fill_(0)

        if init_mode == "xavier_uniform":
            nn.init.xavier_uniform_(self.mlp.weight.data)
            self.mlp.bias.data.fill_(0)
            nn.init.xavier_uniform_(self.output.weight.data)
            self.output.bias.data.fill_(0)

        if init_mode == "kaiming_normal":
            nn.init.kaiming_normal_(self.mlp.weight.data)
            self.mlp.bias.data.fill_(0)
            nn.init.kaiming_normal_(self.output.weight.data)
            self.output.bias.data.fill_(0)

        if init_mode == "kaiming_uniform":
            nn.init.kaiming_uniform_(self.mlp.weight.data)
            self.mlp.bias.data.fill_(0)
            nn.init.kaiming_uniform_(self.output.weight.data)
            self.output.bias.data.fill_(0)