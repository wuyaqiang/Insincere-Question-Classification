#coding=utf-8
import os
import re
import time
import datetime
import random
import pickle
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

from load_word2vec import *

from model.C_LSTM import C_LSTM
from model.LSTM_GRU import LSTM_GRU

class IterMixin(object):
    def __iter__(self):
        for attr, value in self.__dict__.items():
            yield attr, value
class HyperParameter(IterMixin):
    '''
    超参数
    '''
    def __init__(self,
                 use_cuda = False,

                 # 词向量维度
                 embedding_dim = 0,
                 embedding_filepath = '',
                 max_sent_len = 53,
                 max_word_num=95000,

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
        self.use_cuda = use_cuda

        self.embedding_dim = embedding_dim
        self.embedding_filepath = embedding_filepath
        self.max_sent_len = max_sent_len
        self.max_word_num = max_word_num

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

options = HyperParameter(

    use_cuda = torch.cuda.is_available(),

    embedding_dim=300,              # 词向量维度
    max_sent_len = 70,              # 最大句子长度
    max_word_num = 95000,           # 最大词典长度

    hidden_size=128,                # RNN隐藏单元大小
    context_vec_size=128,           # 自注意力上下文向量大小
    mlp1_hidden_size=128,           # 多层感知器隐藏层大小
    mlp2_hidden_size=256,
    kernel_num=128,                 # number of each kind of kernel:
    kernel_sizes="3",               # what kind of kernel sizes

    optim="Adam",
    lr=0.001,                      # default learning rate: Adam_0.001     Adagrad_0.01    RMSprop_0.01
    lr_decay=0.1,                   # default lr_decay: Adagrad_0
    weight_decay=0.0001,                 # default weight_decay: 0
    momentum=0.9,                   # default momentum: 0
    betas=(0.9, 0.999),             # default betas: (0.9, 0.999)
    eps=1e-8,                       # default eps: 1e-08

    gradient_clip=False,            # 是否进行Gradient clipping
    init_mode="xavier_normal",      # 权重初始化方式
    dropout_p=0.2,                  # dropout大小
    class_size=1,                   # 输出类别个数
    epoch=10,                       # epoch大小 循环轮数
    batch_size=512,
    print_interval=200,               # 间隔多少个batch打印一次loss
    early_stopping=True,            # 是否执行early_stopping
    save_model=False,                # 保存模型
    save_mode="best",
    model_path="./saved_model",
)

pre_processing_start = time.time()
# 读取数据
train_df = pd.read_csv("./data/train.csv")
test_df = pd.read_csv("./data/test.csv")

# 设置一系列随机种子，尽量使实验结果可重现
def set_random_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

# 找出最佳分类阈值，以及该阈值下的F1值
def threshold_search(y_true, y_proba):
    best_threshold = 0
    best_score = 0
    for threshold in [i * 0.01 for i in range(100)]:
        score = f1_score(y_true=y_true, y_pred=y_proba > threshold)
        if score > best_score:
            best_threshold = threshold
            best_score = score
    return best_threshold, best_score

def compute_measure(predict, label, thresh):
    '''Compute precision, recall, f1 and accuracy'''
    tn, fp, fn, tp = 0, 0, 0, 0
    for i in range(len(predict)):
        if predict[i] >= thresh and int(label[i]) == 1:
            tp += 1
        elif predict[i] >= thresh and int(label[i]) == 0:
            fp += 1
        elif predict[i] < thresh and int(label[i]) == 1:
            fn += 1
        else:
            tn += 1
    return tp, fp, fn, tn

# 符号
puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£', '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]
# 人工定义一个常见错别字词典：
mis_spell_dict = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have", 'colour': 'color', 'centre': 'center', 'favourite': 'favorite', 'travelling': 'traveling', 'counselling': 'counseling', 'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor', 'organisation': 'organization', 'wwii': 'world war 2', 'citicise': 'criticize', 'youtu ': 'youtube ', 'Qoura': 'Quora', 'sallary': 'salary', 'Whta': 'What', 'narcisist': 'narcissist', 'howdo': 'how do', 'whatare': 'what are', 'howcan': 'how can', 'howmuch': 'how much', 'howmany': 'how many', 'whydo': 'why do', 'doI': 'do I', 'theBest': 'the best', 'howdoes': 'how does', 'mastrubation': 'masturbation', 'mastrubate': 'masturbate', "mastrubating": 'masturbating', 'pennis': 'penis', 'Etherium': 'Ethereum', 'narcissit': 'narcissist', 'bigdata': 'big data', '2k17': '2017', '2k18': '2018', 'qouta': 'quota', 'exboyfriend': 'ex boyfriend', 'airhostess': 'air hostess', "whst": 'what', 'watsapp': 'whatsapp', 'demonitisation': 'demonetization', 'demonitization': 'demonetization', 'demonetisation': 'demonetization'}
# 人工定义需要去掉的停用词：
need_remove_words = ['a', 'to', 'of', 'and']

# 处理语料中的标点符号
def clean_punctuation(sent):
    sent = str(sent)
    for punc in puncts:
        sub_str = " " + punc + " "
        sent = sent.replace(punc, sub_str)
    return sent
# 处理语料中的数字
def clean_number(sent):
    sent = re.sub('[0-9]{5,}', '#####', sent)
    sent = re.sub('[0-9]{4}', '####', sent)
    sent = re.sub('[0-9]{3}', '###', sent)
    sent = re.sub('[0-9]{2}', '##', sent)
    return sent
# 处理语料中的常见错别字, 替换
def clean_mis_spell(sent):
    for key, value in mis_spell_dict.items():
        sent = sent.replace(key, value)
    return sent
# 去停用词
def clean_stopwords(sent):
    sent = [word for word in sent.split() if word not in need_remove_words]
    removed_sent = " ".join(sent)
    return removed_sent
# 去除低频词, freq以下去掉
def clean_low_freq_words(sent, freq, freq_dict):
    sent = [word for word in sent.split() if freq_dict[word] >= freq]
    removed_sent = " ".join(sent)
    return removed_sent
# 统计词表词频
def build_vocab_count(sentences):
    vocab_count = {}
    for sentence in sentences:
        for word in sentence:
            if word in vocab_count:
                vocab_count[word] += 1
            else:
                vocab_count[word] = 1
    return vocab_count


# 对数据做如下预处理：
train_df['question_text'] = train_df['question_text'].str.lower()
test_df['question_text'] = test_df['question_text'].str.lower()

train_df["question_text"] = train_df["question_text"].apply(lambda x: clean_punctuation(x))
test_df["question_text"] = test_df["question_text"].apply(lambda x: clean_punctuation(x))

train_df["question_text"] = train_df["question_text"].apply(lambda x: clean_number(x))
test_df["question_text"] = test_df["question_text"].apply(lambda x: clean_number(x))

train_df["question_text"] = train_df["question_text"].apply(lambda x: clean_mis_spell(x))
test_df["question_text"] = test_df["question_text"].apply(lambda x: clean_mis_spell(x))

train_df["question_text"] = train_df["question_text"].apply(lambda x: clean_stopwords(x))
test_df["question_text"] = test_df["question_text"].apply(lambda x: clean_stopwords(x))

# train_freq_dict = build_vocab_count(train_df["question_text"].apply(lambda x: x.split()))
# test_freq_dict = build_vocab_count(test_df["question_text"].apply(lambda x: x.split()))
# train_df["question_text"] = train_df["question_text"].apply(lambda x: clean_low_freq_words(x, 3, train_freq_dict))
# test_df["question_text"] = test_df["question_text"].apply(lambda x: clean_low_freq_words(x, 3, test_freq_dict))

train_df["question_text"] = train_df["question_text"].fillna("_##_")
test_df["question_text"] = test_df["question_text"].fillna("_##_")

x_train = train_df['question_text'].values
x_test = test_df['question_text'].values

# Tokenize the sentences
tokenizer = Tokenizer(num_words=options.max_word_num, filters='')
tokenizer.fit_on_texts(list(x_train))
x_train = tokenizer.texts_to_sequences(x_train)
x_test = tokenizer.texts_to_sequences(x_test)
# Pad the sentences
x_train = pad_sequences(x_train, maxlen=options.max_sent_len)
x_test = pad_sequences(x_test, maxlen=options.max_sent_len)
# Get the target values
y_train = train_df['target'].values
print('Pre-processing Time: ', time.time()-pre_processing_start)

options.vocab_size = min(options.max_word_num, len(tokenizer.word_index))
options.vocab = tokenizer.word_index

# 读取预训练的词向量
read_pre_trained_start = time.time()
# embedding_dict = load_google_bin("./word embedding/GoogleNews-vectors-negative300.bin", options)
embedding_dict = load_word2vec("./word embedding/glove.840B.300d.txt", options)
# embedding_dict = load_word2vec("./word embedding/paragram_300_sl999.txt", options)
# embedding_dict = load_word2vec("./word embedding/wiki-news-300d-1M.vec", options)

# embedding_dict = pickle.load(open('./word embedding/glove.840B.300d.pickle', 'rb'))
# embedding_dict = np.mean([glove_embedding, paragram_embedding], axis=0)
print('Read Pre-trained Word Embedding: ', time.time()-read_pre_trained_start)
# with open('./word embedding/wiki-news-300d-1M.pickle', 'wb') as f:
#     pickle.dump(embedding_dict, f)

set_random_seed(2018)

def train(epoch_num, train_loader, opt, loss_function):
    model.train(mode=True)

    start_time = time.time()
    total_loss = 0
    all_batch_loss = []

    param = filter(lambda p: p.requires_grad, model.parameters())
    # 选择优化器
    if opt.optim == "SGD":
        optimizer = torch.optim.SGD(
            param,
            lr=opt.lr,
            momentum=opt.momentum,
            weight_decay=opt.weight_decay
        )
    else:
        optimizer = torch.optim.Adam(
            param,
            lr=opt.lr,
            betas=opt.betas,
            eps=opt.eps,
            weight_decay=opt.weight_decay
        )

    for batch_index, (data, label) in enumerate(train_loader):
        if opt.use_cuda:
            data = data.cuda()
            label = label.cuda()
        model.batch_size = len(data)

        predict = model(data)
        loss = loss_function(predict, label)
        # model.zero_grad()
        optimizer.zero_grad()

        loss.backward()

        if opt.gradient_clip == True:
            # Gradient clipping in case of gradient explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

        optimizer.step()

        all_batch_loss.append([epoch_num, loss.item()])
        total_loss += loss.item()

        if batch_index % opt.print_interval == 0 and batch_index > 0:
            cur_loss = total_loss / opt.print_interval
            elapsed_time = time.time() - start_time
            print('| Epoch {:2d} | {:4d}/{:3d} batches | {:5.2f} s/batch | '
                  'loss {:5.4f} |'.format(epoch_num, batch_index, len(train_loader),
                                          elapsed_time / opt.print_interval, cur_loss))
            total_loss = 0
            start_time = time.time()

    return all_batch_loss
def evaluate(x_val_fold, y_val_fold, valid_loader, opt, loss_function):
    model.eval()
    total_loss = 0
    valid_preds = np.zeros((len(x_val_fold)))

    for i, (data, label) in enumerate(valid_loader):
        if opt.use_cuda:
            data = data.cuda()
            label = label.cuda()
        model.batch_size = len(data)

        predict = model(data)
        loss = loss_function(predict, label)
        total_loss += loss.item()
        valid_preds[i * opt.batch_size: (i+1) * opt.batch_size] = predict.cpu().data.numpy()[:, 0]

    f1 = f1_score(y_true=y_val_fold.cpu().numpy()[:, 0], y_pred=valid_preds > 0.5)
    batch_mean_loss = total_loss / len(valid_loader)  # 每个batch的平均loss

    return f1, batch_mean_loss, valid_preds

x_test = torch.tensor(x_test, dtype=torch.long)
if options.use_cuda:
    x_test = x_test.cuda()
test = TensorDataset(x_test)
test_loader = DataLoader(test, batch_size=options.batch_size, shuffle=False)
test_prediction = np.zeros((len(test_df)))
train_prediction = np.zeros((len(train_df)))

splits = list(StratifiedKFold(n_splits=4, shuffle=True, random_state=2018).split(x_train, y_train))

init_lr = options.lr  # 记录下初始学习率, 因为训练过程中会变
for i, (train_idx, valid_idx) in enumerate(splits):
    x_train_fold = torch.tensor(x_train[train_idx], dtype=torch.long)
    y_train_fold = torch.tensor(y_train[train_idx, np.newaxis], dtype=torch.float32)
    x_val_fold = torch.tensor(x_train[valid_idx], dtype=torch.long)
    y_val_fold = torch.tensor(y_train[valid_idx, np.newaxis], dtype=torch.float32)
    if options.use_cuda:
        x_train_fold = x_train_fold.cuda()
        y_train_fold = y_train_fold.cuda()
        x_val_fold = x_val_fold.cuda()
        y_val_fold = y_val_fold.cuda()
    options.lr = init_lr

    # model = LSTM_GRU(vocab_size, embedding_matrix, options)
    model = C_LSTM(embedding_dict, options)

    if options.use_cuda:
        model = model.cuda()
    # loss_fn = nn.BCEWithLogitsLoss(reduction='sum')
    loss_fn = nn.BCELoss(reduction='sum')

    train_dataset = TensorDataset(x_train_fold, y_train_fold)
    valid_dataset = TensorDataset(x_val_fold, y_val_fold)

    train_loader = DataLoader(train_dataset, batch_size=options.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=options.batch_size, shuffle=False)

    print('-'*30, ' Fold ', i+1, ' ', '-'*30)

    local_time = time.strftime("%Y-%m-%d-%Hh%Mm", time.localtime())

    training_loss = []              # 记录训练过程中每个batch的loss
    f1_valuate = []                 # 记录每个每个epoch之后的验证集F1值
    mean_epoch_loss_val = []        # 记录每个epoch之后的验证集loss
    early_stopped = False           # 记录是否发生了early stopping

    for epoch in range(1, options.epoch + 1):

        print("| Learning Rate: ", options.lr)

        epoch_start = time.time()
        all_batch_loss = train(epoch, train_loader, options, loss_fn)
        epoch_end = time.time()
        epoch_time = str(datetime.timedelta(seconds=(epoch_end - epoch_start))).split(".")[0]

        f1_val, mean_loss_val, valid_preds_fold = evaluate(x_val_fold, y_val_fold, valid_loader, options, loss_fn)
        print('-' * 30)
        print('| Epoch {:2d}   F1: {:.4f}   valid loss: {:.4f}   time: {}'
              .format(epoch, f1_val, mean_loss_val, epoch_time))
        print('-' * 30)

        f1_valuate.append(f1_val)
        mean_epoch_loss_val.append(mean_loss_val)
        training_loss += all_batch_loss

        model_state_dict = model.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'setting': options,
            'epoch': epoch
        }
        if options.save_model:
            if options.save_mode == 'all':
                model_path = options.model_path + '/' + local_time
                model_name = model_path + '/f1_{f1:3.3f}_epoch_at_{epoch}.chkpt'.format(f1=100 * f1_val,
                                                                                        epoch=epoch)
                if not os.path.exists(model_path):
                    os.makedirs(model_path)
                torch.save(checkpoint, model_name)
            elif options.save_mode == 'best':
                model_path = options.model_path + '/' + local_time
                model_name = model_path + '/best_f1_{f1:3.3f}_epoch_at_{epoch}.chkpt'.format(f1=100 * f1_val,
                                                                                             epoch=epoch)
                if f1_val >= max(f1_valuate):
                    if not os.path.exists(model_path):
                        os.makedirs(model_path)
                    torch.save(checkpoint, model_name)
                    print('| Information | The checkpoint file has been updated |')

        if epoch > 1 and mean_epoch_loss_val[-1] > mean_epoch_loss_val[-2]:
            # 一旦验证集loss开始出现上升或波动, 立即将learning rate降低
            options.lr = options.lr * 0.5
            # early_stopped = True
            # break
        if epoch > 2 and options.early_stopping == True:
            if mean_epoch_loss_val[-1] > mean_epoch_loss_val[-2] \
                    and mean_epoch_loss_val[-2] > mean_epoch_loss_val[-3]:
                early_stopped = True
                break
    if early_stopped:
        print('\nEarly Stopping from Fold ', i)

    # predict all samples in the test set batch per batch
    test_pred_fold = np.zeros((len(test_df)))
    for i, (x_batch,) in enumerate(test_loader):
        model.batch_size = len(x_batch)
        y_pred = model(x_batch).detach()
        test_pred_fold[i*options.batch_size : (i+1)*options.batch_size] = y_pred.cpu().data.numpy()[:,0]
    test_prediction += test_pred_fold / len(splits)
    train_prediction[valid_idx] = valid_preds_fold

best_threshold, best_f1 = threshold_search(y_train, train_prediction)
print('Best Threshold: ', best_threshold, ' Best F1: ', best_f1)
sub = test_df[['qid']].copy()
sub['prediction'] = test_prediction > best_threshold
sub.to_csv('submission.csv', index=False)









