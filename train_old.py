#coding=utf-8
import datetime
import os
import pickle
import time
import numpy as np
import torch
import torch.nn as nn

from utils.utility import compute_measure
from utils.hyper_param import HyperParameter
from get_data import get_training_data
from model.model_1 import BiLSTM
from model.model_2 import CNN
from model.model_5 import LSTM_CNN
# from model.model_6 import C_LSTM

def data_package(data, vocab, requires_grad=False):
    '''
    将训练数据中的所有词, 根据其对应词典中的序号, 转换成数值向量, 并做好padding
    '''
    max_length = 0
    for sent in data:
        max_length = max(max_length, len(sent))
    # 设置最大句子长度, 超过就截断:
    max_length = min(max_length, 53)

    batch_data = np.zeros((len(data), max_length), dtype=int)
    for sent_idx, sent in enumerate(data):
        for word_idx, word in enumerate(sent[ : max_length]):
            if word not in vocab:
                batch_data[sent_idx, word_idx] = vocab['<unk>']
            else:
                batch_data[sent_idx, word_idx] = vocab[word]
    packaged_data = torch.from_numpy(batch_data)
    # para_data.requires_grad_(requires_grad)
    return packaged_data

def train(epoch_num, x_train_text, y_train, vocab, opt):
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
    elif opt.optim == "Adagrad":
        optimizer = torch.optim.Adagrad(
            param,
            lr=opt.lr,
            lr_decay=opt.lr_decay,
            weight_decay=opt.weight_decay
        )
    elif opt.optim == "RMSprop":
        optimizer = torch.optim.RMSprop(
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

    for batch_index, start_index in enumerate(range(0, len(x_train_text), opt.batch_size)):
        data = data_package(x_train_text[start_index: start_index + opt.batch_size], vocab)
        label = torch.FloatTensor(y_train[start_index: start_index + opt.batch_size]).reshape(-1, 1)
        if use_cuda:
            data = data.cuda()
            label = label.cuda()

        model.zero_grad()
        optimizer.zero_grad()

        predict = model(data)
        loss = loss_function(predict, label)
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
            print('| epoch {:2d} | {:4d}/{:3d} batches | {:5.2f} s/batch | '
                  'loss {:5.4f} |'.format(epoch_num, batch_index, len(x_train_text) // opt.batch_size,
                                          elapsed_time / opt.print_interval, cur_loss))
            total_loss = 0
            start_time = time.time()

    return all_batch_loss

def evaluate(epoch_num, x_val_text, y_val, vocab, opt):
    '''
    计算验证集上的precision,recall,以及F1值
    '''
    model.eval()  # 从训练模式切换到验证模式
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    total_loss = 0
    loss_function = nn.BCELoss()
    with torch.no_grad():
        # for thresh in np.arange(0.2, 0.62, 0.02):
        #     for batch_index, start_index in enumerate(range(0, len(x_val_text), opt.batch_size)):
        #         data = data_package(x_val_text[start_index: start_index + opt.batch_size], vocab)
        #         label = torch.FloatTensor(y_val[start_index: start_index + opt.batch_size]).reshape(-1, 1)
        #         if use_cuda:
        #             data = data.cuda()
        #             label = label.cuda()
        #
        #         predict = model(data)
        #         loss = loss_function(predict, label)
        #         total_loss += loss.item()
        #
        #         # for i in range(5):
        #         #     print(predict[i].cpu().data.numpy()[0], '----', label[i].cpu().data.numpy()[0])
        #
        #         batch_tp, batch_fp, batch_fn, batch_tn = compute_measure(predict.cpu().data, label.cpu().data, thresh)
        #         TP += batch_tp
        #         FP += batch_fp
        #         FN += batch_fn
        #         TN += batch_tn
        #
        #     pre = TP / float(FP + TP + 1e-8)
        #     rec = TP / float(FN + TP + 1e-8)
        #     f1 = 2 * pre * rec / (pre + rec + 1e-8)
        #     acc = (TN + TP) / float(TN + FP + FN + TP + 1e-8)
        #
        #     # print('-' * 80)
        #     print('Threshold ', format(thresh, '0.2f'), '   ',
        #           'F1:', format(f1, '0.5f'), '  ',
        #           'Precision:', format(pre, '0.5f'), '  ',
        #           'Recall:', format(rec, '0.5f'))

        for batch_index, start_index in enumerate(range(0, len(x_val_text), opt.batch_size)):
            data = data_package(x_val_text[start_index: start_index + opt.batch_size], vocab)
            label = torch.FloatTensor(y_val[start_index: start_index + opt.batch_size]).reshape(-1, 1)
            if use_cuda:
                data = data.cuda()
                label = label.cuda()

            predict = model(data)
            loss = loss_function(predict, label)
            total_loss += loss.item()

            # for i in range(5):
            #     print(predict[i].cpu().data.numpy()[0], '----', label[i].cpu().data.numpy()[0])

            batch_tp, batch_fp, batch_fn, batch_tn = compute_measure(predict.cpu().data,
                                                                     label.cpu().data,
                                                                     thresh=0.5)
            TP += batch_tp
            FP += batch_fp
            FN += batch_fn
            TN += batch_tn

        pre = TP / float(FP + TP + 1e-8)
        rec = TP / float(FN + TP + 1e-8)
        f1 = 2 * pre * rec / (pre + rec + 1e-8)
        acc = (TN + TP) / float(TN + FP + FN + TP + 1e-8)

        print('-' * 80)
        print('Precision:', pre)
        print('Recall:', rec)
        print('F1:', f1)
        print('Acc:', acc)

    batch_mean_loss = total_loss / (len(x_val_text) / opt.batch_size)  # 每个batch的平均loss
    # batch_mean_loss = total_loss    # 整个一轮的loss
    val_metric = [epoch_num, pre, rec, f1, acc, batch_mean_loss]

    return f1, batch_mean_loss, val_metric

if __name__ == '__main__':

    use_cuda = torch.cuda.is_available()

    options = HyperParameter(

        # 词向量维度
        embedding_dim=300,

        # embedding_filepath='./word embedding/GoogleNews-vectors-negative300.pickle',
        embedding_filepath='./word embedding/glove.840B.300d.pickle',
        # embedding_filepath='./word embedding/paragram_300_sl999.pickle',
        # embedding_filepath='./word embedding/wiki-news-300d-1M.pickle',

        # RNN隐藏单元大小
        hidden_size=256,

        # 自注意力上下文向量大小
        context_vec_size=128,

        # 多层感知器隐藏层大小
        mlp1_hidden_size=128,
        mlp2_hidden_size=256,

        # number of each kind of kernel:
        kernel_num=256,
        # what kind of kernel sizes:
        kernel_sizes="2",

        # dropout大小
        dropout_p=0.2,

        # 输出类别个数
        class_size=1,

        # 优化算法
        optim="Adam",
        lr=0.00005,  # default learning rate: Adam_0.001     Adagrad_0.01    RMSprop_0.01
        lr_decay=0.1,  # default lr_decay: Adagrad_0
        weight_decay=0,  # default weight_decay: 0
        momentum=0.9,  # default momentum: 0
        betas=(0.9, 0.999),  # default betas: (0.9, 0.999)
        eps=1e-8,  # default eps: 1e-08

        # 是否进行Gradient clipping
        gradient_clip=False,

        # 权重初始化方式
        init_mode="xavier_normal",

        # epoch大小 循环轮数
        epoch=20,
        batch_size=256,

        # 间隔多少个batch打印一次loss
        print_interval=50,

        # 是否执行early_stopping
        early_stopping=True,

        # 保存模型
        save_model=True,
        save_mode="best",
        model_path="./saved_model",
    )

    print('> 数据预处理开始 <')
    time_2 = time.time()
    x_train_text, y_train, x_val_text, y_val, vocab = get_training_data(options.batch_size)
    vocab_size = len(vocab)

    print('> 总样例个数: ', len(y_train) + len(y_val))
    print('> 训练集大小: ', len(y_train))
    print('> 验证集大小: ', len(y_val))
    print('> 正样例个数: ', sum(y_train) + sum(y_val))
    print('> 正样例占比:  %.2f%%' % (((sum(y_train) + sum(y_val)) / (len(y_train) + len(y_val))) * 100))
    print('> 数据预处理结束, 所用时间: ', time.time() - time_2)

    print('> 开始读取预训练词向量 <')
    time_1 = time.time()
    embedding_dict = pickle.load(open(options.embedding_filepath, 'rb'))
    print('> 成功读取预存储的embedding_dict <')
    print('> 词表大小: ', vocab_size)
    print('> 预训练包含的词个数: ', len(embedding_dict))
    print('> 读取预训练词向量结束, 所用时间: ', time.time() - time_1)

    print('> 模型开始训练 <')
    train_start_time = time.time()


    # model = BiLSTM(vocab_size, embedding_dict, vocab, use_cuda, options)
    # model = CNN(vocab_size, embedding_dict, vocab, use_cuda, options)
    model = LSTM_CNN(vocab_size, embedding_dict, vocab, use_cuda, options)
    # model = C_LSTM(vocab_size, embedding_dict, vocab, use_cuda, options)

    if use_cuda:
        model = model.cuda()

    loss_function = nn.BCELoss()
    if use_cuda:
        loss_function = loss_function.cuda()

    local_time = time.strftime("%Y-%m-%d-%Hh%Mm", time.localtime())

    init_lr = options.lr # 记录下初始学习率, 因为训练过程中会变
    training_loss = []  # 记录训练过程中每个batch的loss
    val_metrics = []  # 记录每个epoch之后验证集的结果: [epoch, precision, recall, f1, acc, valid loss]
    f1_valuate = []  # 记录每个每个epoch之后的验证集F1值
    mean_epoch_loss_val = []  # 记录每个epoch之后的验证集loss
    early_stopped = False  # 记录是否发生了early stopping

    try:
        for epoch in range(1, options.epoch + 1):

            print("| Learning Rate: ", options.lr)

            epoch_start = time.time()
            all_batch_loss = train(epoch, x_train_text, y_train, vocab, options)
            epoch_end = time.time()
            f1_val, mean_loss_val, metric = evaluate(epoch, x_val_text, y_val, vocab, options)

            epoch_time = str(datetime.timedelta(seconds=(epoch_end - epoch_start))).split(".")[0]
            print('| end of epoch {:2d} | time: {} | valid loss {:5.8f} | '
                  .format(epoch, epoch_time, mean_loss_val))

            f1_valuate.append(f1_val)
            mean_epoch_loss_val.append(mean_loss_val)
            val_metrics.append(metric)
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
            #
            # if epoch > 3 and options.early_stopping == True:
            #     if mean_epoch_loss_val[-1] > mean_epoch_loss_val[-2] \
            #             and mean_epoch_loss_val[-2] > mean_epoch_loss_val[-3] \
            #             and mean_epoch_loss_val[-3] > mean_epoch_loss_val[-4]:
            #         early_stopped = True
            #         break
            print('-' * 80)

        if early_stopped:
            print('\nEarly Stopping from Training.')
        # print('Evaluating Model On Test Set... ')
        # f1_val, mean_loss_val, test_metric = evaluate(epoch, x_test_text, y_test, vocab, options)
        # print('| End of training | F1: {:5.5f} | Test Loss {:5.8f} | '.format(f1_val, mean_loss_val))
        # print('-' * 80)

    except KeyboardInterrupt:
        print(' ')
        print('Early Stopping from Training by KeyboardInterrupt.')
        # print('Evaluating Model On Test Set... ')
        # f1_val, mean_loss_val, test_metric = evaluate(x_test_text, y_test, sent_position_test, word_position_test, word_dict,
        #                                      options)
        # print('| End of training | F1: {:5.5f} | Test Loss {:5.8f} | '.format(f1_val, mean_loss_val))
        # print('-' * 80)

    training_time = str(datetime.timedelta(seconds=(time.time() - train_start_time))).split(".")[0]
    print('> Training is Over, The Whole Time: ', training_time)

    # 记录实验日志:
    # save_metrics(training_loss, val_metrics, test_metric, local_time, options, init_lr)
    # plot_metrics(training_loss, val_metrics, test_metric, local_time)



















