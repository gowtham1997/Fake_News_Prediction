from keras.utils import to_categorical
import numpy as np

import tensorflow as tf
from nltk.stem import   WordNetLemmatizer
from tqdm import tqdm_notebook
tf.enable_eager_execution()

wordnet_lemmatizer = WordNetLemmatizer()


def get_suffixed_columns(df, suffix):
    return [col for col in df.columns if col.endswith(suffix)]


def convert_col_to_list(df, column):
    df[column] = df[column].map(
        lambda x: np.array([int(i) for i in x.split(',')]))
    return df


def encode_meta_features(df_batch, meta_columns, global_df):
    encoded_subject_id = np.array(df_batch['subject_list_id'].to_list())
    # print(encoded_subject_id.shape)
    encoded_meta_cols = []
    for col in meta_columns:
        if col == 'subject_list_id':
            continue
        encoded_meta_col = to_categorical(
            df_batch[col], global_df[col].max() + 1)
        # print(encoded_meta_col.shape)
        encoded_meta_cols.append(encoded_meta_col)

    return np.hstack((encoded_subject_id, ) + tuple(encoded_meta_cols))


def encode_labels(labels, num_classes):

    encoded_labels = to_categorical(labels, num_classes)
    return encoded_labels


def tensorise_batch(batch):
    return tf.convert_to_tensor(batch)


def batch(df, batch_size, repeat_first_batch=False):
    idx = 0
    meta_columns = get_suffixed_columns(df, '_id')
    counts_columns = get_suffixed_columns(df, '_counts')
    text_columns = ['statement', 'justification']

    while True:
        # repeat first batch to overfit and check if the network works ok
        if repeat_first_batch:
            batch_df = df[0: 0 + batch_size]
        else:
            batch_df = df[idx: idx + batch_size]

        statements_batch = batch_df[text_columns[0]]
        justifications_batch = batch_df[text_columns[1]]

        meta_batch = batch_df[meta_columns]
        counts_batch = batch_df[counts_columns]
        # print(meta_batch.shape)
        # print(meta_batch)

        meta_features = encode_meta_features(meta_batch, meta_columns, df)
        counts_features = counts_batch.values
        # print(meta_features.shape)

        binary_labels = encode_labels(batch_df['binary_label'],
                                      num_classes=2)
        labels = encode_labels(batch_df['label'], num_classes=6)

        statements_tf = tensorise_batch(statements_batch)
        justifications_tf = tensorise_batch(justifications_batch)

        meta_tf = tensorise_batch(meta_features)
        counts_tf = tensorise_batch(counts_features)

        binary_labels_tf = tensorise_batch(binary_labels)
        labels_tf = tensorise_batch(labels)

        yield statements_tf, justifications_tf, meta_tf, counts_tf, \
            binary_labels_tf, labels_tf

        idx += batch_size
        if (len(df) - idx) < batch_size or idx >= len(df):
            raise StopIteration


# https://colab.research.google.com/github/tensorflow/tensorflow/blob/master/tensorflow/contrib/eager/python/examples/nmt_with_attention/nmt_with_attention.ipynb
class ColumnIndex():
    def __init__(self, sentences):
        self.sentences = sentences
        self.word2idx = {}
        self.idx2word = {}
        self.vocab = set()

        self.create_index()

    def create_index(self):
        for phrase in self.sentences:
            self.vocab.update(phrase.split(' '))

        self.vocab = sorted(self.vocab)[:10000]

        self.word2idx['<pad>'] = 0
        for index, word in enumerate(self.vocab):
            self.word2idx[word] = index + 1

        for word, index in self.word2idx.items():
            self.idx2word[index] = word


def max_length(tensor):
    return max(len(t) for t in tensor)


def prepare_text_column(df, column, max_len, glove_emb_dict):
    sentences = df[column].astype('str').values

    # remove stop words
    mod_sentences = []
    for sen in sentences:
        word_list = [wordnet_lemmatizer.lemmatize(token) for token in sen.split(' ')]
        mod_sentences.append(' '.join(word_list))
        
    sentences = [' '.join(sen.split(' ')[:max_len]) for sen in mod_sentences]

    col_utils = ColumnIndex(sentences)

    input_tensor = [[col_utils.word2idx.get(s, len(col_utils.word2idx))
                     for s in sen.split(' ')] for sen in sentences]

    vocab_size = len(col_utils.word2idx)
    print(f'vocab_size for {column} is {vocab_size}')

    embedding_matrix = np.zeros((vocab_size, 300))


    for word, i in tqdm_notebook(col_utils.word2idx.items()):
        embedding_vector = glove_emb_dict.get(word, None)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

        input_tensor = tf.keras.preprocessing.sequence.pad_sequences(
            input_tensor,
            maxlen=max_len,
            padding='post')

    return input_tensor, col_utils, embedding_matrix


def prepare_glove_dict(path):
    # load the whole embedding into memory
    embeddings_index = dict()
    f = open(path)
    for line in tqdm_notebook(f):
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    return embeddings_index
