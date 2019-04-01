# standard imports
import time
import random
import pickle
import os
import re
import numpy as np
import pandas as pd

# pytorch imports
import torch
import torch.nn as nn
import torch.utils.data

# imports for preprocessing the questions
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# cross validation and metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

# train_df = pd.read_csv("./data/train.csv")
# test_df = pd.read_csv("./data/test.csv")

def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
# seed_everything()

def threshold_search(y_true, y_proba):
    best_threshold = 0
    best_score = 0
    for threshold in [i * 0.01 for i in range(100)]:
        score = f1_score(y_true=y_true, y_pred=y_proba > threshold)
        if score > best_score:
            best_threshold = threshold
            best_score = score
    return best_threshold, best_score

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

embed_size = 300 # how big is each word vector
max_features = 95000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 70 # max number of words in a question to use

puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£', '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]
mis_spell_dict = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have", 'colour': 'color', 'centre': 'center', 'favourite': 'favorite', 'travelling': 'traveling', 'counselling': 'counseling', 'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor', 'organisation': 'organization', 'wwii': 'world war 2', 'citicise': 'criticize', 'youtu ': 'youtube ', 'Qoura': 'Quora', 'sallary': 'salary', 'Whta': 'What', 'narcisist': 'narcissist', 'howdo': 'how do', 'whatare': 'what are', 'howcan': 'how can', 'howmuch': 'how much', 'howmany': 'how many', 'whydo': 'why do', 'doI': 'do I', 'theBest': 'the best', 'howdoes': 'how does', 'mastrubation': 'masturbation', 'mastrubate': 'masturbate', "mastrubating": 'masturbating', 'pennis': 'penis', 'Etherium': 'Ethereum', 'narcissit': 'narcissist', 'bigdata': 'big data', '2k17': '2017', '2k18': '2018', 'qouta': 'quota', 'exboyfriend': 'ex boyfriend', 'airhostess': 'air hostess', "whst": 'what', 'watsapp': 'whatsapp', 'demonitisation': 'demonetization', 'demonitization': 'demonetization', 'demonetisation': 'demonetization'}

def clean_punctuation(sent):
    sent = str(sent)
    for punc in puncts:
        sub_str = " " + punc + " "
        sent = sent.replace(punc, sub_str)
    return sent
def clean_number(sent):
    sent = re.sub('[0-9]{5,}', '#####', sent)
    sent = re.sub('[0-9]{4}', '####', sent)
    sent = re.sub('[0-9]{3}', '###', sent)
    sent = re.sub('[0-9]{2}', '##', sent)
    return sent
def clean_mis_spell(sent):
    for key, value in mis_spell_dict.items():
        sent = sent.replace(key, value)
    return sent

# train_df["question_text"] = train_df["question_text"].str.lower()
# test_df["question_text"] = test_df["question_text"].str.lower()
#
# train_df["question_text"] = train_df["question_text"].apply(lambda x: clean_punctuation(x))
# test_df["question_text"] = test_df["question_text"].apply(lambda x: clean_punctuation(x))
#
# train_df["question_text"] = train_df["question_text"].apply(lambda x: clean_number(x))
# test_df["question_text"] = test_df["question_text"].apply(lambda x: clean_number(x))
#
# train_df["question_text"] = train_df["question_text"].apply(lambda x: clean_mis_spell(x))
# test_df["question_text"] = test_df["question_text"].apply(lambda x: clean_mis_spell(x))
#
# with open("./data/train_df.pickle", "wb") as f:
#     pickle.dump(train_df, f)
# with open("./data/test_df.pickle", "wb") as f:
#     pickle.dump(test_df, f)
train_df = pickle.load(open('./data/train_df.pickle', 'rb'))
test_df = pickle.load(open('./data/test_df.pickle', 'rb'))

print('Train Data Size: ', len(train_df))
print('Test Data Size: ', len(test_df))

# fill up the missing values
x_train = train_df["question_text"].fillna("_##_").values
x_test = test_df["question_text"].fillna("_##_").values

# Tokenize the sentences
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(x_train))
x_train = tokenizer.texts_to_sequences(x_train)
x_test = tokenizer.texts_to_sequences(x_test)

# Pad the sentences
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)

# Get the target values
y_train = train_df['target'].values

def load_glove(word_index):
    EMBEDDING_FILE = './word embedding/glove.840B.300d.txt'

    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype='float32')[:300]

    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))

    all_embs = np.stack(embeddings_index.values())
    emb_mean, emb_std = -0.005838499, 0.48782197
    embed_size = all_embs.shape[1]

    # word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector

    return embedding_matrix

def load_fasttext(word_index):
    EMBEDDING_FILE = './word embedding/wiki-news-300d-1M.vec'

    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype='float32')

    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE) if len(o) > 100)

    all_embs = np.stack(embeddings_index.values())
    emb_mean, emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]

    # word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector

    return embedding_matrix

def load_para(word_index):
    EMBEDDING_FILE = './word embedding/paragram_300_sl999.txt'

    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype='float32')

    embeddings_index = dict(
        get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding="utf8", errors='ignore') if len(o) > 100)

    all_embs = np.stack(embeddings_index.values())
    emb_mean, emb_std = -0.0053247833, 0.49346462
    embed_size = all_embs.shape[1]

    # word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector

    return embedding_matrix

# missing entries in the embedding are set using np.random.normal so we have to seed here too
# seed_everything()

time1 = time.time()
# glove_embeddings = load_glove(tokenizer.word_index)
# paragram_embeddings = load_para(tokenizer.word_index)
# embedding_matrix = np.mean([glove_embeddings, paragram_embeddings], axis=0)
# with open('./word embedding/embedding_matrix.pickle', 'wb') as f:
#     pickle.dump(embedding_matrix, f)
embedding_matrix = pickle.load(open('./word embedding/embedding_matrix.pickle', 'rb'))
print('Word2Vec Loading Time: ', time.time()-time1)

splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=10).split(x_train, y_train))

class Attention(nn.Module):
    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super(Attention, self).__init__(**kwargs)

        self.supports_masking = True

        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.features_dim = 0

        weight = torch.zeros(feature_dim, 1)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight)

        if bias:
            self.b = nn.Parameter(torch.zeros(step_dim))

    def forward(self, x, mask=None):
        feature_dim = self.feature_dim
        step_dim = self.step_dim

        eij = torch.mm(
            x.contiguous().view(-1, feature_dim),
            self.weight
        ).view(-1, step_dim)

        if self.bias:
            eij = eij + self.b

        eij = torch.tanh(eij)
        a = torch.exp(eij)

        if mask is not None:
            a = a * mask

        a = a / torch.sum(a, 1, keepdim=True) + 1e-10

        weighted_input = x * torch.unsqueeze(a, -1)
        return torch.sum(weighted_input, 1)

class C_LSTM(nn.Module):

    def __init__(self):
        super(C_LSTM, self).__init__()

        self.hidden_size = 64
        self.in_channels = 1
        self.out_channels = 128
        self.kernel_sizes = [2]

        self.dropout = nn.Dropout(0.3)
        self.embedding = nn.Embedding(max_features, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        self.layer_norm_embed = nn.LayerNorm(300)

        self.lstm = nn.LSTM(self.out_channels, self.hidden_size, bidirectional=True, batch_first=True)
        self.gru = nn.GRU(300, self.hidden_size, bidirectional=True, batch_first=True)
        self.layer_norm_rnn = nn.LayerNorm(self.hidden_size * 2)

        self.lstm_attention = Attention(self.hidden_size * 2, maxlen-1)
        self.gru_attention = Attention(self.hidden_size * 2, maxlen)

        self.convolutions = nn.ModuleList([nn.Conv2d(self.in_channels, self.out_channels,
                                                     kernel_size=(k, 300)) for k in self.kernel_sizes])
        self.layer_norm_cnn = nn.LayerNorm(self.out_channels)

        self.mlp = nn.Linear(self.hidden_size * 6, self.hidden_size)
        self.output = nn.Linear(self.hidden_size, 1)
        self.relu = nn.ReLU()
        self.init_weights()

    def forward(self, batch_data):
        embed = self.embedding(batch_data)
        # embed = self.layer_norm_embed(embed)
        word_embed = embed.unsqueeze(1)   # (batch_size, in_channels, sent_length, embedding_dim)
        word_conved = [self.relu(conv(word_embed)).squeeze(3) for conv in self.convolutions]
        word_conved = word_conved[0]
        word_conved = torch.transpose(word_conved, 1, 2)
        # word_conved = self.layer_norm_cnn(word_conved)
        lstm_output, _ = self.lstm(word_conved)
        gru_output, _ = self.gru(embed)

        lstm_atten = self.lstm_attention(lstm_output)
        gru_atten = self.gru_attention(gru_output)
        avg_pool = torch.mean(lstm_output, 1)
        max_pool, _ = torch.max(lstm_output, 1)

        feature_vec = torch.cat((gru_atten, avg_pool, max_pool), 1)
        # feature_vec = self.layer_norm_rnn(feature_vec)
        mlp_output = self.dropout(self.relu(self.mlp(feature_vec)))
        output = self.output(mlp_output)
        return output

    def init_weights(self):
        nn.init.xavier_normal_(self.mlp.weight.data)
        nn.init.xavier_normal_(self.output.weight.data)

class LSTM_GRU(nn.Module):
    def __init__(self):
        super(LSTM_GRU, self).__init__()

        hidden_size = 64

        self.embedding = nn.Embedding(max_features, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False

        self.embedding_dropout = nn.Dropout2d(0.1)
        self.lstm = nn.LSTM(embed_size, hidden_size, bidirectional=True, batch_first=True)
        self.gru = nn.GRU(hidden_size * 2, hidden_size, bidirectional=True, batch_first=True)

        self.lstm_attention = Attention(hidden_size * 2, maxlen)
        self.gru_attention = Attention(hidden_size * 2, maxlen)

        self.linear = nn.Linear(hidden_size * 8, hidden_size * 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.out = nn.Linear(hidden_size * 2, 1)

        self.init_weights()

    def forward(self, x):
        h_embedding = self.embedding(x)
        h_embedding = torch.squeeze(
            self.embedding_dropout(torch.unsqueeze(h_embedding, 0)))

        h_lstm, _ = self.lstm(h_embedding)
        h_gru, _ = self.gru(h_lstm)

        h_lstm_atten = self.lstm_attention(h_lstm)
        h_gru_atten = self.gru_attention(h_gru)

        # global average pooling
        avg_pool = torch.mean(h_gru, 1)
        # global max pooling
        max_pool, _ = torch.max(h_gru, 1)

        conc = torch.cat((h_lstm_atten, h_gru_atten, avg_pool, max_pool), 1)
        conc = self.relu(self.linear(conc))
        conc = self.dropout(conc)
        out = self.out(conc)

        return out

    def init_weights(self):
        nn.init.xavier_normal_(self.linear.weight.data)
        nn.init.xavier_normal_(self.out.weight.data)

batch_size = 512 # how many samples to process at once
n_epochs = 6 # how many times to iterate over all samples

# matrix for the out-of-fold predictions
train_preds = np.zeros((len(train_df)))
# matrix for the predictions on the test set
test_preds = np.zeros((len(test_df)))

# always call this before training for deterministic results
# seed_everything()

x_test_cuda = torch.tensor(x_test, dtype=torch.long).cuda()
test = torch.utils.data.TensorDataset(x_test_cuda)
test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)

for i, (train_idx, valid_idx) in enumerate(splits):
    # split data in train / validation according to the KFold indeces
    # also, convert them to a torch tensor and store them on the GPU (done with .cuda())
    x_train_fold = torch.tensor(x_train[train_idx], dtype=torch.long).cuda()
    y_train_fold = torch.tensor(y_train[train_idx, np.newaxis], dtype=torch.float32).cuda()
    x_val_fold = torch.tensor(x_train[valid_idx], dtype=torch.long).cuda()
    y_val_fold = torch.tensor(y_train[valid_idx, np.newaxis], dtype=torch.float32).cuda()

    # model = LSTM_GRU()
    model = C_LSTM()
    # make sure everything in the model is running on the GPU
    model.cuda()

    # define binary cross entropy loss
    # note that the model returns logit to take advantage of the log-sum-exp trick
    # for numerical stability in the loss
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.002)

    train = torch.utils.data.TensorDataset(x_train_fold, y_train_fold)
    valid = torch.utils.data.TensorDataset(x_val_fold, y_val_fold)

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=False)

    print('Fold ', i + 1)

    for epoch in range(n_epochs):
        # set train mode of the model. This enables operations which are only applied during training like dropout
        start_time = time.time()
        model.train()
        avg_loss = 0.
        for x_batch, y_batch in train_loader:
            # Forward pass: compute predicted y by passing x to the model.
            y_pred = model(x_batch)
            # Compute and print loss.
            loss = loss_fn(y_pred, y_batch)
            # Before the backward pass, use the optimizer object to zero all of the
            # gradients for the Tensors it will update (which are the learnable weights
            # of the model)
            optimizer.zero_grad()
            # Backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # Calling the step function on an Optimizer makes an update to its parameters
            optimizer.step()
            avg_loss += loss.item() / len(train_loader)

        # set evaluation mode of the model. This disabled operations which are only applied during training like dropout
        model.eval()

        # predict all the samples in y_val_fold batch per batch
        valid_preds_fold = np.zeros((x_val_fold.size(0)))
        test_preds_fold = np.zeros((len(test_df)))

        avg_val_loss = 0.
        for i, (x_batch, y_batch) in enumerate(valid_loader):
            y_pred = model(x_batch).detach()

            avg_val_loss += loss_fn(y_pred, y_batch).item() / len(valid_loader)
            valid_preds_fold[i * batch_size:(i + 1) * batch_size] = sigmoid(y_pred.cpu().numpy())[:, 0]

        valid_f1 = f1_score(y_true=y_val_fold.cpu().numpy()[:, 0], y_pred=valid_preds_fold > 0.33)
        elapsed_time = time.time() - start_time
        print('Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f}\tval_f1={:.4f} \t time={:.2f}s'.format(
            epoch + 1, n_epochs, avg_loss, avg_val_loss, valid_f1, elapsed_time))

    # predict all samples in the test set batch per batch
    for i, (x_batch,) in enumerate(test_loader):
        y_pred = model(x_batch).detach()

        test_preds_fold[i * batch_size:(i + 1) * batch_size] = sigmoid(y_pred.cpu().numpy())[:, 0]

    train_preds[valid_idx] = valid_preds_fold
    test_preds += test_preds_fold / len(splits)

best_threshold, best_score = threshold_search(y_train, train_preds)

submission = test_df[['qid']].copy()
submission['prediction'] = test_preds > best_threshold
submission.to_csv('submission.csv', index=False)