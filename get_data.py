#coding=utf-8
import sys
import codecs
import time
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pre_processing import read_google_bin_file
from pre_processing import clean_punctuation, clean_mis_spell, clean_number, clean_stopwords, clean_low_freq_words
from vocab_process import build_vocab, build_vocab_count, Instance


def get_instance(data_path, file_existed):
    if file_existed == True:
        all_instance = pickle.load(open('./data/all_instance.pickle', 'rb'))
    else:
        all_instance = []
        embedding_dict = read_google_bin_file()
        data_df = pd.read_csv(data_path)
        # 缺失值填充
        data_df["question_text"].fillna("_na_", inplace=True)

        data_df["question_text"] = data_df["question_text"].apply(lambda x: clean_punctuation(x, embedding_dict))
        data_df["question_text"] = data_df["question_text"].apply(lambda x: clean_number(x))
        data_df["question_text"] = data_df["question_text"].apply(lambda x: clean_mis_spell(x))
        data_df["question_text"] = data_df["question_text"].apply(lambda x: clean_stopwords(x))
        freq_dict = build_vocab_count(data_df["question_text"].apply(lambda x: x.split()))
        data_df["question_text"] = data_df["question_text"].apply(lambda x: clean_low_freq_words(x, 3, freq_dict))

        for index, row in data_df.iterrows():
            if len(row["question_text"].split()) == 0:
                continue
            instance = Instance(row['question_text'], row['target'])
            all_instance.append(instance)
        with codecs.open('./data/all_instance.pickle', 'wb') as f:
            pickle.dump(all_instance, f)
    return all_instance

def get_training_data(batch_size):
    all_instance = pickle.load(open('./data/all_instance.pickle', 'rb'))

    vocab = build_vocab(all_instance)

    train, val = train_test_split(all_instance, test_size=0.1, train_size=0.9,
                                  random_state=2018, shuffle=True)

    # 测试样例数不被BATCH_SIZE整除时, 去掉最后一个不足一个batch的数据, 将这部分数据放到训练集里
    if (len(val) % batch_size) != 0:
        redundant_num = len(val) % batch_size
        train += val[-redundant_num : ]
        val = val[ : -redundant_num]

    # 训练样例数不被BATCH_SIZE整除时, 则通过采样将最后一个batch填充满.
    if (len(train) % batch_size) != 0:
        padding_num = batch_size - (len(train) % batch_size)
        sample_index = np.random.choice(len(train), size=padding_num, replace=False)
        for index in sample_index:
            train.append(train[index])

    x_train_text = []
    x_val_text = []
    y_train = []
    y_val = []
    for item in train:
        x_train_text.append(item.text.split())
        y_train.append(item.label)
    for item in val:
        x_val_text.append(item.text.split())
        y_val.append(item.label)

    return x_train_text, y_train, x_val_text, y_val, vocab


if __name__ == '__main__':

    # project_path = sys.path[1]
    # train_data_path = project_path + "/data/train.csv"
    #
    # all_instance = get_instance(train_data_path, file_existed=False)
    #
    # vocab = build_vocab(all_instance)
    # print('词表大小: ', len(vocab))

    # train, val = get_training_data(128)
    # print(len(train))
    # print(len(val))
    # for i in train[:10]:
    #     print(i.label, ' ', i.text)
    # print('\n')
    # for i in val[:10]:
    #     print(i.label, ' ', i.text)

    pass