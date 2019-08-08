import numpy as np
import pandas as pd

import utils
import nnlm_eager as nnlm
import models

import tensorflow as tf
tf.enable_eager_execution()

TRAIN_CSV_PATH = 'LIAR-PLUS/dataset/mod_train.csv'
VAL_CSV_PATH = 'LIAR-PLUS/dataset/mod_val.csv'
TEST_CSV_PATH = 'LIAR-PLUS/dataset/mod_test.csv'

EMBEDDING_DIR = 'pretrained_embeddings/'


MAX_LEN_STATMENT = 20
MAX_LEN_JUSTIFICATION = 20

LSTM_HIDDEN_UNITS_STATEMENT = 256
LSTM_HIDDEN_UNITS_JUSTIFICATION = 256

META_MODEL_UNITS = 128
COUNT_MODEL_UNITS = 128

EMBEDDING_DIM = 128

SENTENCE_ENCODER = 'NNLM'


if __name__ == "__main__":

    train_df = pd.read_csv(TRAIN_CSV_PATH)
    val_df = pd.read_csv(VAL_CSV_PATH)
    test_df = pd.read_csv(TEST_CSV_PATH)

    meta_columns = utils.get_suffixed_columns(train_df, '_id')
    counts_columns = utils.get_suffixed_columns(train_df, '_counts')
    text_columns = ['statement', 'justification']

    train_text_df = train_df[text_columns]
    val_text_df = val_df[text_columns]
    test_text_df = test_df[text_columns]

    train_meta_df = train_df[meta_columns]
    val_meta_df = val_df[meta_columns]
    test_meta_df = test_df[meta_columns]

    train_counts_df = train_df[counts_columns]
    val_counts_df = val_df[counts_columns]
    test_counts_df = test_df[counts_columns]

    train_meta_features = utils.encode_meta_features(
        train_meta_df, meta_columns)
    val_meta_features = utils.encode_meta_features(
        val_meta_df, meta_columns)
    test_meta_features = utils.encode_meta_features(
        test_meta_df, meta_columns)

    train_count_features = np.hstack(train_counts_df.values)
    val_count_features = np.hstack(val_counts_df.values)
    test_count_features = np.hstack(test_counts_df.values)

    meta_features_shape = (None, train_meta_features.shape[-1])
    count_features_shape = (None, train_count_features.shape[-1])

    if SENTENCE_ENCODER == 'NNLM':
        encoder = nnlm.NNLMEncoder(EMBEDDING_DIR + 'nnlm', EMBEDDING_DIM)

    meta_model = models.MetaModel(META_MODEL_UNITS)
    count_model = models.CountModel(COUNT_MODEL_UNITS)
