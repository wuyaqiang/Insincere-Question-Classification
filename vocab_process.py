#coding=utf-8
import string

class Instance:
    def __init__(self, text, label):
        self.text = text
        self.label = label

# 建立词表
def build_vocab(sentences):
    # sentences: list of list of words
    # return: dictionary of words in training corpus
    # if file_existed == True:
    #     vocab = pickle.load(open('./data/vocabulary.pickle', 'rb'))
    # else:
    vocab = {'<pad>': 0, '<unk>': 1}
    for sentence in sentences:
        for word in sentence.text.split():
            if word not in vocab:
                vocab[word] = len(vocab)
    # with codecs.open('./data/vocabulary.pickle', 'wb') as f:
    #     pickle.dump(vocab, f)
    return vocab

# 统计词表词频
def build_vocab_count(sentences):
    # sentences: list of list of words
    # return: dictionary of words and their count
    # if file_existed == True:
    #     vocab_count = pickle.load(open('./data/vocabulary_freq.pickle', 'rb'))
    # else:
    vocab_count = {}
    for sentence in sentences:
        for word in sentence:
            if word in vocab_count:
                vocab_count[word] += 1
            else:
                vocab_count[word] = 1
    return vocab_count

# 统计未登录词
def check_out_of_vocab(vocab_count, embedding_dict):
    exist_word = {}
    oov_word = {}
    exist_count = 0    # 包含在word2vec中的词频的总数
    oov_count = 0      # 未登录词的词频总数
    for word in vocab_count:
        if word in embedding_dict:
            exist_word[word] = embedding_dict[word]
            exist_count += vocab_count[word]
        else:
            oov_word[word] = vocab_count[word]
            oov_count += vocab_count[word]
    print('{:.2%} words of vocab founded in word2vec.'.format(len(exist_word) / len(vocab_count)))
    print('{:.2%} words of all text founded in word2vec.'.format(exist_count / (exist_count + oov_count)))
    sorted_oov_word = sorted(oov_word.items(), key=lambda x: x[1])[::-1]
    return sorted_oov_word

# 统计未登录punctuation
def check_out_of_punctuation(embedding_dict):
    exist_punc = ""     # 有预训练的词向量的符号
    oov_punc = ""       # 没有预训练词向量的符号
    all_punctuation = string.punctuation
    # print('all punctuations: ', all_punctuation)
    for punc in all_punctuation:
        if punc in embedding_dict:
            exist_punc += punc
        else:
            oov_punc += punc
    return exist_punc, oov_punc

if __name__ == '__main__':
    pass
    # all_instance = pickle.load(open('./data/all_instance.pickle', 'rb'))
    # sentences = []
    # for item in all_instance:
    #     sentences.append(item.text.split())
    # print(sentences[1])
    # vocab_count = build_vocab_count(sentences)
    # word_embedding_path = './word embedding/GoogleNews-vectors-negative300.bin'
    # embedding_dict = KeyedVectors.load_word2vec_format(word_embedding_path, binary=True)
    # sorted_oov_word = check_out_of_vocab(vocab_count, embedding_dict)
    # print(sorted_oov_word[:20])




















