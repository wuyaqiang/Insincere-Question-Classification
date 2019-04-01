#coding=utf-8
'''
模型三: BERT Model
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class BERT(nn.Module):
    
    def __init__(self, vocab_size, embedding_dict, vocab, use_cuda=False, opt=None):
        super(BERT, self).__init__()

        self.use_cuda = use_cuda
        self.vocab = vocab
        self.embedding_dict = embedding_dict
        self.vocab_size = vocab_size
        self.embed_size = opt.embedding_dim

