#coding=utf-8
'''
C-LSTM 结构 (来自论文 A C-LSTM Neural Network for Text Classification )
'''
import torch
import torch.nn as nn
from .Attention import Attention

class C_LSTM(nn.Module):

    def __init__(self, embedding_dict, opt=None):
        super(C_LSTM, self).__init__()

        self.use_cuda = opt.use_cuda
        self.vocab = opt.vocab
        self.embedding_dict = embedding_dict
        self.vocab_size = opt.vocab_size
        self.embed_size = opt.embedding_dim
        self.batch_size = opt.batch_size
        self.hidden_size = opt.hidden_size
        self.context_vec_size = opt.context_vec_size
        self.mlp1_hidden_size = opt.mlp1_hidden_size
        self.output_size = opt.class_size
        self.in_channels = 1
        self.out_channels = opt.kernel_num

        self.dropout = nn.Dropout(p=opt.dropout_p)
        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)
        self.layer_norm_embed = nn.LayerNorm(self.embed_size)

        self.rnn = nn.LSTM(self.out_channels, self.hidden_size, bidirectional=True, batch_first=True)
        self.layer_norm_rnn = nn.LayerNorm(self.hidden_size * 2)

        self.S1 = nn.Linear(self.hidden_size * 2, self.context_vec_size, bias=False)
        self.S2 = nn.Linear(self.context_vec_size, 1, bias=False)

        self.kernel_sizes = [int(i) for i in opt.kernel_sizes.split(",")]
        self.convolutions = nn.ModuleList([nn.Conv2d(self.in_channels, self.out_channels,
                                                     kernel_size=(k, self.embed_size)) for k in self.kernel_sizes])
        self.layer_norm_cnn = nn.LayerNorm(self.out_channels)

        self.mlp1 = nn.Linear(self.hidden_size * 2, self.mlp1_hidden_size)
        self.mlp2 = nn.Linear(self.mlp1_hidden_size, self.output_size)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.init_weights()
        self.init_pretrained_embedding()

    def forward(self, batch_data):
        embed = self.dropout(self.embedding(batch_data))
        # embed = self.layer_norm_embed(embed)

        word_embed = embed.unsqueeze(1)   # (batch_size, in_channels, sent_length, embedding_dim)
        word_conved = [self.relu(conv(word_embed)).squeeze(3) for conv in self.convolutions]
        word_conved = word_conved[0]
        word_conved = torch.transpose(word_conved, 1, 2)
        # word_conved = self.layer_norm_cnn(word_conved)

        hidden = self.init_hidden(self.batch_size)
        output, hidden = self.rnn(word_conved, hidden)
        # output = self.layer_norm_rnn(output)

        # 自注意力模块:
        H = output
        h1 = self.S1(H)
        h1 = self.tanh(h1)
        h2 = self.S2(h1)
        attention_weight = self.softmax(h2)
        attention_weight = torch.transpose(attention_weight, 1, 2)
        attention_output = torch.matmul(attention_weight, H).squeeze()

        feature_vec = attention_output
        # feature_vec = self.layer_norm_rnn(feature_vec)
        # feature_vec = torch.transpose(hidden[0], 0, 1).contiguous().view(self.batch_size, -1)

        mlp1_output = self.dropout(self.relu(self.mlp1(feature_vec)))
        mlp2_output = self.mlp2(mlp1_output)
        predict = self.sigmoid(mlp2_output)

        return predict

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

    def init_pretrained_embedding(self):
        # 初始化预训练的词向量矩阵
        nn.init.normal_(self.embedding.weight.data)
        self.embedding.weight.data[0].fill_(0)
        loaded_count = 0
        for word, idx in self.vocab.items():
            if word not in self.embedding_dict:
                continue
            if idx < self.vocab_size:
                self.embedding.weight.data[idx] = torch.from_numpy(self.embedding_dict[word]).view(-1)
                loaded_count += 1
        print('> %d words from pre-trained word vectors loaded.' % loaded_count)

    def init_weights(self):
        nn.init.xavier_normal_(self.mlp1.weight.data)
        self.mlp1.bias.data.fill_(0)
        nn.init.xavier_normal_(self.mlp2.weight.data)
        self.mlp2.bias.data.fill_(0)
        nn.init.xavier_normal_(self.S1.weight.data)
        nn.init.xavier_normal_(self.S2.weight.data)