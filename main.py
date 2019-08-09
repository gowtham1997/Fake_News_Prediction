import pandas as pd

import utils
import nnlm_eager as nnlm
import models
import train
import time
import os

from tensorflow.keras import optimizers

import tensorflow as tf
tf.enable_eager_execution()

# import warnings
# warnings.filterwarnings('ignore')

TRAIN_CSV_PATH = 'LIAR-PLUS/dataset/mod_train.csv'
VAL_CSV_PATH = 'LIAR-PLUS/dataset/mod_val.csv'
TEST_CSV_PATH = 'LIAR-PLUS/dataset/mod_test.csv'

EMBEDDING_DIR = 'pretrained_embeddings/'


MAX_LEN_STATMENT = 50
MAX_LEN_JUSTIFICATION = 100

LSTM_HIDDEN_UNITS_STATEMENT = 256
LSTM_HIDDEN_UNITS_JUSTIFICATION = 256

META_MODEL_UNITS = 256
COUNTS_MODEL_UNITS = 64
MULTIMODE_MODEL_UNITS = 1024

EMBEDDING_DIM = 128

SENTENCE_ENCODER = 'NNLM'
# SENTENCE_ENCODER = 'USC'
# SENTENCE_ENCODER = 'GLOVE'

BINARY_CLASSIFICATION = False

if BINARY_CLASSIFICATION:
    NUM_CLASSES = 2
    CHECKPOINT_PATH = 'checkpoints_binary/'
else:
    NUM_CLASSES = 6
    CHECKPOINT_PATH = 'checkpoints_multiclass/'

NUM_EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 0.001

TRAIN_TEXT_ENCODER = True
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
                                            MULTIMODE_MODEL_UNITS,
                                            NUM_CLASSES)
    optimizer = optimizers.Adam(learning_rate=LEARNING_RATE)

    checkpoint_dir = CHECKPOINT_PATH
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    if TRAIN_TEXT_ENCODER:
        # somehow tfeager models are not trackable
        # hence we cant save the embedding
        # ValueError: `Checkpoint` was expecting a trackable object (an object derived from `TrackableBase`), got <nnlm_eager.NNLMEncoder object at 0x7f689064bda0>. If you believe this object should be trackable (i.e. it is part of the TensorFlow Python API and manages state), please open an issue.
        checkpoint = tf.train.Checkpoint(
            optimizer=optimizer,
            # text_encoder=text_encoder,
            meta_model=meta_model,
            counts_model=counts_model,
            multimode_model=multimode_model)
    else:
        checkpoint = tf.train.Checkpoint(
            optimizer=optimizer,
            meta_model=meta_model,
            counts_model=counts_model,
            multimode_model=multimode_model)

    best_val_acc = 0.00
    for epoch in range(NUM_EPOCHS):
        start = time.time()
        loss, accuracy = train.train_one_epoch(
            train_df, BATCH_SIZE, optimizer,
            text_encoder, meta_model, counts_model, multimode_model,
            train_text_encoder=TRAIN_TEXT_ENCODER,
            repeat_first_batch=REPEAT_FIRST_BATCH,
            binary_classification=BINARY_CLASSIFICATION)
        print(
            f'Epoch: {epoch + 1}, '
            f'train_loss: {loss}, train_accuracy: {accuracy}')
        val_loss, val_accuracy = train.predict_one_epoch(
            val_df, BATCH_SIZE, text_encoder, meta_model, counts_model,
            multimode_model,
            binary_classification=BINARY_CLASSIFICATION)
        print(
            f'Epoch: {epoch + 1}, '
            f'  val_loss: {val_loss},   val_accuracy: {val_accuracy}')
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            checkpoint.save(file_prefix=checkpoint_prefix)
            print('Model saved !!!')

        test_loss, test_accuracy = train.predict_one_epoch(
            test_df, BATCH_SIZE, text_encoder, meta_model, counts_model,
            multimode_model,
            binary_classification=BINARY_CLASSIFICATION)
        print(f'  Test_loss: {test_loss},   Test_accuracy: {test_accuracy}')

        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

    # restore latest checkpoint
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    test_loss, test_accuracy = train.predict_one_epoch(
        test_df, BATCH_SIZE, text_encoder, meta_model, counts_model,
        multimode_model,
        binary_classification=BINARY_CLASSIFICATION)
    print(f'  Test_loss: {test_loss},   Test_accuracy: {test_accuracy}')
