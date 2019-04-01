#coding=utf-8
import torch
import torch.nn as nn
from model.Attention import Attention

class LSTM_GRU(nn.Module):
    def __init__(self, vocab_size, embedding_matrix, opt):
        super(LSTM_GRU, self).__init__()
        self.use_cuda = opt.use_cuda
        self.hidden_size = opt.hidden_size
        self.mlp1_hidden_size = opt.mlp1_hidden_size

        self.dropout = nn.Dropout(opt.dropout_p)
        self.embedding = nn.Embedding(vocab_size, opt.embedding_dim)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))

        self.lstm = nn.LSTM(300, self.hidden_size, bidirectional=True, batch_first=True)
        self.gru = nn.GRU(self.hidden_size * 2, self.hidden_size, bidirectional=True, batch_first=True)
        self.lstm_attention = Attention(self.hidden_size * 2, opt.max_sent_len)
        self.gru_attention = Attention(self.hidden_size * 2, opt.max_sent_len)

        self.mlp = nn.Linear(self.hidden_size * 8, self.mlp1_hidden_size)
        self.output = nn.Linear(self.mlp1_hidden_size, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embed = self.dropout(self.embedding(x))

        lstm_output, _ = self.lstm(embed)
        gru_output, _ = self.gru(lstm_output)

        lstm_atten_output = self.lstm_attention(lstm_output)
        gru_atten_output = self.gru_attention(gru_output)

        avg_pool = torch.mean(gru_output, 1)
        max_pool, _ = torch.max(gru_output, 1)

        concat_vec = torch.cat((lstm_atten_output, gru_atten_output, avg_pool, max_pool), 1)
        mlp_output = self.relu(self.mlp(concat_vec))
        mlp_output = self.dropout(mlp_output)
        output = self.output(mlp_output)
        # output = self.sigmoid(output)

        return output