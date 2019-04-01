#coding=utf-8
import re
import pickle
import codecs
import numpy as np
from gensim.models import KeyedVectors

from vocab_process import check_out_of_punctuation, Instance, build_vocab
from relate_dict import mis_spell_dict, need_remove_words

# 处理语料中的标点符号
def clean_punctuation(sent, embedding_dict):
    exist_punc, oov_punc = check_out_of_punctuation(embedding_dict)
    sent = str(sent)
    for punc in "/-":
        sent = sent.replace(punc, ' ')
    for punc in exist_punc:
        sent = sent.replace(punc, ' '+punc+' ')
    for punc in oov_punc + "”“‘’":
        sent = sent.replace(punc, '')
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

# 读取Google预训练的词向量word2vec, 二进制文件
def read_google_bin_file():
    word_embedding_path = './word embedding/GoogleNews-vectors-negative300.bin'
    embedding_dict = KeyedVectors.load_word2vec_format(word_embedding_path, binary=True)
    return embedding_dict

def read_corpus_related_pretrain_wordvec(filepath):
    # 将整个语料词典中的词的词向量预先读出来存起来, 就不用每次再读整个预训练的词向量文件了
    embedding_dict = {}
    with codecs.open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.split()
            if len(line) > 0:
                word = line[0]
                if len(line) == 301:
                    embed_vec = np.array(line[1:], dtype="float32")
                    embedding_dict[word] = embed_vec

    all_instance = pickle.load(open('./data/all_instance.pickle', 'rb'))
    vocab = build_vocab(all_instance)
    print('语料词典大小: ', len(vocab))

    corpus_embed_dict = {}
    for word in vocab:
        if word in embedding_dict:
            corpus_embed_dict[word] = embedding_dict[word]
    print('读取了多少个词: ', len(corpus_embed_dict))
    # with open('./word embedding/GoogleNews-vectors-negative300.pickle', 'wb') as f:
    # with open('./word embedding/glove.840B.300d.pickle', 'wb') as f:
    # with open('./word embedding/paragram_300_sl999.pickle', 'wb') as f:
    with open('./word embedding/wiki-news-300d-1M.pickle', 'wb') as f:
        pickle.dump(corpus_embed_dict, f)

# 统计语料中的所有句子长度
def count_sent_length(all_instance):
    length_dict = {}
    for item in all_instance:
        length = len(item.text.split())
        if length in length_dict:
            length_dict[length] += 1
        else:
            length_dict[length] = 1
    sorted_length = sorted(length_dict.items(), key=lambda x: x[0])
    return sorted_length


if __name__ == '__main__':
    # all_instance = pickle.load(open('./data/all_instance.pickle', 'rb'))
    #
    # sent_length = count_sent_length(all_instance)
    # for i in sent_length:
    #     print(i)

    # read_corpus_related_pretrain_wordvec('./word embedding/GoogleNews-vectors-negative300.txt')
    # read_corpus_related_pretrain_wordvec('./word embedding/glove.840B.300d.txt')
    # read_corpus_related_pretrain_wordvec('./word embedding/paragram_300_sl999.txt')
    # read_corpus_related_pretrain_wordvec('./word embedding/wiki-news-300d-1M.vec')

    pass



















