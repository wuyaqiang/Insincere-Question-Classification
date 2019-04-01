#coding=utf-8
'''
模型二: CNN
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    
    def __init__(self, vocab_size, embedding_dict, vocab, use_cuda=False, opt=None):
        super(CNN, self).__init__()

        self.use_cuda = use_cuda
        self.vocab = vocab
        self.embedding_dict = embedding_dict
        self.vocab_size = vocab_size
        self.embed_size = opt.embedding_dim

        self.in_channels = 1
        self.out_channels = opt.kernel_num
        self.kernel_sizes = [int(i) for i in opt.kernel_sizes.split(",")]

        self.dropout = nn.Dropout(p=opt.dropout_p)
        self.embedding = nn.Embedding(vocab_size, self.embed_size)

        # 卷积层:
        self.convolutions = nn.ModuleList([nn.Conv2d(self.in_channels, self.out_channels,
                                                     kernel_size=(k, self.embed_size)) for k in self.kernel_sizes])
        self.relu = nn.ReLU()

        # 两个全连接层:
        self.mlp1 = nn.Linear(len(self.kernel_sizes) * self.out_channels, opt.mlp1_hidden_size)
        self.output = nn.Linear(opt.mlp1_hidden_size, opt.class_size)
        self.sigmoid = nn.Sigmoid()

        self.init_weights(init_mode=opt.init_mode)
        self.init_pretrained_embedding()


    def forward(self, data):
        word_embed = self.embedding(data)
        word_embed = self.dropout(word_embed)

        word_embed = word_embed.unsqueeze(1)   # (batch_size, in_channels, sent_length, embedding_dim)
        word_conved = [self.relu(conv(word_embed)).squeeze(3) for conv in self.convolutions]
        word_pooled = [F.max_pool1d(item, item.size(2)).squeeze(2) for item in word_conved]
        word_concated = torch.cat(word_pooled, 1)

        mlp_output = self.mlp1(word_concated)
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

    def init_weights(self, init_mode):

        if init_mode == "normal":
            nn.init.normal_(self.mlp1.weight.data)
            self.mlp1.bias.data.fill_(0)
            nn.init.normal_(self.output.weight.data)
            self.output.bias.data.fill_(0)

        if init_mode == "uniform":
            nn.init.uniform_(self.mlp1.weight.data)
            self.mlp1.bias.data.fill_(0)
            nn.init.uniform_(self.output.weight.data)
            self.output.bias.data.fill_(0)

        if init_mode == "xavier_normal":
            nn.init.xavier_normal_(self.mlp1.weight.data)
            self.mlp1.bias.data.fill_(0)
            nn.init.xavier_normal_(self.output.weight.data)
            self.output.bias.data.fill_(0)

        if init_mode == "xavier_uniform":
            nn.init.xavier_uniform_(self.mlp1.weight.data)
            self.mlp1.bias.data.fill_(0)
            nn.init.xavier_uniform_(self.output.weight.data)
            self.output.bias.data.fill_(0)

        if init_mode == "kaiming_normal":
            nn.init.kaiming_normal_(self.mlp1.weight.data)
            self.mlp1.bias.data.fill_(0)
            nn.init.kaiming_normal_(self.output.weight.data)
            self.output.bias.data.fill_(0)

        if init_mode == "kaiming_uniform":
            nn.init.kaiming_uniform_(self.mlp1.weight.data)
            self.mlp1.bias.data.fill_(0)
            nn.init.kaiming_uniform_(self.output.weight.data)
            self.output.bias.data.fill_(0)



























