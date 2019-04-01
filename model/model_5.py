#coding=utf-8
'''
模型三:
LSTM 和 CNN 结合：将LSTM + self-attention得到的特征向量，和CNN得到的特征向量，直接concatenation
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM_CNN(nn.Module):

    def __init__(self, vocab_size, embedding_dict, vocab, use_cuda=False, opt=None):
        super(LSTM_CNN, self).__init__()

        self.use_cuda = use_cuda
        self.vocab = vocab
        self.embedding_dict = embedding_dict
        self.vocab_size = vocab_size
        self.embed_size = opt.embedding_dim
        self.batch_size = opt.batch_size
        self.hidden_size = opt.hidden_size
        self.context_vec_size = opt.context_vec_size
        self.output_size = opt.class_size

        self.dropout = nn.Dropout(p=opt.dropout_p)
        self.embedding = nn.Embedding(vocab_size, self.embed_size)
        self.rnn = nn.LSTM(self.embed_size, self.hidden_size, bidirectional=True, batch_first=True)

        self.S1 = nn.Linear(self.hidden_size * 2, self.context_vec_size, bias=False)
        self.S2 = nn.Linear(self.context_vec_size, 1, bias=False)

        self.in_channels = 1
        self.out_channels = opt.kernel_num
        self.kernel_sizes = [int(i) for i in opt.kernel_sizes.split(",")]
        self.convolutions = nn.ModuleList([nn.Conv2d(self.in_channels, self.out_channels,
                                                     kernel_size=(k, self.embed_size)) for k in self.kernel_sizes])

        rnn_output_dim = self.hidden_size * 2
        self.position_wise_ffn_1 = PositionwiseFeedForward(rnn_output_dim, rnn_output_dim * 2, opt.dropout_p)
        cnn_output_dim = len(self.kernel_sizes) * self.out_channels
        self.position_wise_ffn_2 = PositionwiseFeedForward(cnn_output_dim, cnn_output_dim * 2, opt.dropout_p)

        self.mlp = nn.Linear(rnn_output_dim + cnn_output_dim, self.output_size)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        self.init_weights()
        self.init_pretrained_embedding()

    def forward(self, batch_data):
        embed = self.dropout(self.embedding(batch_data))

        hidden = self.init_hidden(self.batch_size)
        output, hidden = self.rnn(embed, hidden)
        # last_hidden_output = hidden[0].transpose(0, 1).contiguous()
        # last_hidden_output = last_hidden_output.view(last_hidden_output.size(0), 1, -1).squeeze()

        # 自注意力模块:
        H = output
        h1 = self.S1(H)
        h1 = self.tanh(h1)
        h2 = self.S2(h1)
        attention_weight = self.softmax(h2)
        attention_weight = torch.transpose(attention_weight, 1, 2)
        attention_output = torch.matmul(attention_weight, H).squeeze()

        # rnn_output = torch.cat((last_hidden_output, attention_output), 1)
        rnn_ffn_output = self.position_wise_ffn_1(attention_output)

        word_embed = embed.unsqueeze(1)   # (batch_size, in_channels, sent_length, embedding_dim)
        word_conved = [self.relu(conv(word_embed)).squeeze(3) for conv in self.convolutions]
        word_pooled = [F.max_pool1d(item, item.size(2)).squeeze(2) for item in word_conved]
        word_concated = torch.cat(word_pooled, 1)

        cnn_ffn_output = self.position_wise_ffn_2(word_concated)

        feature_vec = torch.cat((rnn_ffn_output, cnn_ffn_output), 1)
        predict = self.sigmoid(self.mlp(feature_vec))

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

    def init_weights(self):
        nn.init.xavier_normal_(self.S1.weight.data)
        nn.init.xavier_normal_(self.S2.weight.data)
        nn.init.xavier_normal_(self.mlp.weight.data)
        self.mlp.bias.data.fill_(0)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, input_dim, inner_dim, dropout=0.2):
        super(PositionwiseFeedForward, self).__init__()

        self.mlp_1 = nn.Linear(input_dim, inner_dim)
        self.mlp_2 = nn.Linear(inner_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def forward(self, x):
        return self.mlp_2(self.dropout(F.relu(self.mlp_1(x))))

    def init_weights(self):
        nn.init.xavier_normal_(self.mlp_1.weight.data)
        self.mlp_1.bias.data.fill_(0)
        nn.init.xavier_normal_(self.mlp_2.weight.data)
        self.mlp_2.bias.data.fill_(0)