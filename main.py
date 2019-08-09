import pandas as pd

import utils
import nnlm_eager as nnlm
import models
import train

from tensorflow.keras import optimizers

import tensorflow as tf
tf.enable_eager_execution()

# import warnings
# warnings.filterwarnings('ignore')

TRAIN_CSV_PATH = 'LIAR-PLUS/dataset/mod_train.csv'
VAL_CSV_PATH = 'LIAR-PLUS/dataset/mod_val.csv'
TEST_CSV_PATH = 'LIAR-PLUS/dataset/mod_test.csv'

EMBEDDING_DIR = 'pretrained_embeddings/'


MAX_LEN_STATMENT = 20
MAX_LEN_JUSTIFICATION = 20

LSTM_HIDDEN_UNITS_STATEMENT = 256
LSTM_HIDDEN_UNITS_JUSTIFICATION = 256

META_MODEL_UNITS = 128
COUNTS_MODEL_UNITS = 64

EMBEDDING_DIM = 128

SENTENCE_ENCODER = 'NNLM'

BINARY_CLASSIFICATION = True

if BINARY_CLASSIFICATION:
    NUM_CLASSES = 2
else:
    NUM_CLASSES = 6

NUM_EPOCHS = 20
BATCH_SIZE = 8
LEARNING_RATE = 0.001

TRAIN_TEXT_ENCODER = False
REPEAT_FIRST_BATCH = False

if __name__ == "__main__":

    train_df = pd.read_csv(TRAIN_CSV_PATH)
    val_df = pd.read_csv(VAL_CSV_PATH)
    test_df = pd.read_csv(TEST_CSV_PATH)

    train_df = utils.convert_col_to_list(train_df, 'subject_list_id')
    val_df = utils.convert_col_to_list(val_df, 'subject_list_id')
    test_df = utils.convert_col_to_list(test_df, 'subject_list_id')

    # just batch to get shape info
    dummy_gen = utils.batch(train_df, 8, REPEAT_FIRST_BATCH)
    _, _, meta_feats, counts_feats, _, _ = next(dummy_gen)

    meta_features_shape = (None, meta_feats.shape[-1])
    counts_features_shape = (None, counts_feats.shape[-1])
    multimode_features_shape = (
        None, EMBEDDING_DIM + meta_feats.shape[-1] + counts_feats.shape[-1])

    print(meta_features_shape, counts_features_shape, multimode_features_shape)

    if SENTENCE_ENCODER == 'NNLM':
        text_encoder = nnlm.NNLMEncoder(EMBEDDING_DIR + 'nnlm', EMBEDDING_DIM)

    meta_model = models.MetaModel(meta_features_shape, META_MODEL_UNITS)
    counts_model = models.CountModel(counts_features_shape, COUNTS_MODEL_UNITS)
    multimode_model = models.MultiModeModel(multimode_features_shape,
                                            NUM_CLASSES)
    optimizer = optimizers.Adam(learning_rate=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
        loss, accuracy = train.train_one_epoch(
            train_df, BATCH_SIZE, optimizer,
            text_encoder, meta_model, counts_model, multimode_model,
            train_text_encoder=TRAIN_TEXT_ENCODER,
            repeat_first_batch=REPEAT_FIRST_BATCH,
            binary_classification=BINARY_CLASSIFICATION)
        print(
            f'Epoch: {epoch + 1}, '
            f'train_loss: {loss}, train_accuracy: {accuracy}')
        loss, accuracy = train.predict_one_epoch(
            val_df, BATCH_SIZE, text_encoder, meta_model, counts_model,
            multimode_model,
            binary_classification=BINARY_CLASSIFICATION)
        print(
            f'Epoch: {epoch + 1}, '
            f'  val_loss: {loss},   val_accuracy: {accuracy}')
