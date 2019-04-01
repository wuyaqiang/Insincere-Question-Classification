#coding=utf-8
import codecs
import numpy as np
from gensim.models import KeyedVectors

def load_word2vec(file_path, options):
    embedding_dict = {}
    with codecs.open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.split()
            if len(line) == options.embedding_dim + 1:
                word = line[0]
                if word in options.vocab:
                    vec = np.array(line[1:], dtype='float32')
                    embedding_dict[word] = vec
    return embedding_dict

def load_google_bin(file_path, options):
    embedding_dict = KeyedVectors.load_word2vec_format(file_path, binary=True)
    return embedding_dict

def load_glove(word_index, options):
    EMBEDDING_FILE = './word embedding/glove.840B.300d.txt'

    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype='float32')[:300]

    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))

    all_embs = np.stack(embeddings_index.values())
    emb_mean, emb_std = -0.005838499, 0.48782197
    embed_size = all_embs.shape[1]

    # word_index = tokenizer.word_index
    nb_words = min(options.max_word_num, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= options.max_word_num: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector

    return embedding_matrix
def load_fasttext(word_index, options):
    EMBEDDING_FILE = './word embedding/wiki-news-300d-1M.vec'

    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype='float32')

    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE) if len(o) > 100)

    all_embs = np.stack(embeddings_index.values())
    emb_mean, emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]

    # word_index = tokenizer.word_index
    nb_words = min(options.max_word_num, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= options.max_word_num: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector

    return embedding_matrix
def load_para(word_index, options):
    EMBEDDING_FILE = './word embedding/paragram_300_sl999.txt'

    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype='float32')

    embeddings_index = dict(
        get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding="utf8", errors='ignore') if len(o) > 100)

    all_embs = np.stack(embeddings_index.values())
    emb_mean, emb_std = -0.0053247833, 0.49346462
    embed_size = all_embs.shape[1]

    # word_index = tokenizer.word_index
    nb_words = min(options.max_word_num, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= options.max_word_num: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector

    return embedding_matrix