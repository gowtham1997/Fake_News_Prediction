"""Useful functions to train the model

Attributes:
    flags (tf.app.flags): similar to argparse
    FLAGS (tf.app.flags.FlAGS): similar to argparse.parse_args
"""
import sklearn.metrics as metrics
import numpy as np
import warnings
import sklearn.exceptions as exp
import utils
# from tqdm import tqdm
import typing as typ
import os
# to disable tf warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf
tf.enable_eager_execution()
# ignore sklearn's warnings
warnings.filterwarnings("ignore", category=exp.UndefinedMetricWarning)
flags = tf.app.flags
FLAGS = flags.FLAGS

# keeping batchnorm momentum less(than default 0.99) makes sure the val loss is
# stable throughout training
# NOTE: Also while training, the batchnorm layer placed inside Keras sequential
# gives an index(out of range error) while placing it outside works
BATCH_NORM_MOMENTUM = 0.9


def compute_loss(logits, labels,
                 binary_classification=True):
    """Computes multiclass multilabel sigmoid cross entropy loss

    Args:
        logits (TYPE): logits
        labels (TYPE): labels

    Returns:
        TYPE: averaged loss
    """
    if binary_classification:
        loss_fn = tf.keras.backend.binary_crossentropy
    else:
        loss_fn = tf.keras.losses.categorical_crossentropy
    logits = tf.cast(logits, tf.float32)
    labels = tf.cast(labels, tf.float32)

    per_class_loss = loss_fn(
        labels, logits, from_logits=True)

    return tf.reduce_mean(per_class_loss, axis=-1)


def compute_accuracy(labels, logits, binary_classification=True):
    if binary_classification:
        activation = tf.nn.sigmoid
    else:
        activation = tf.nn.softmax

    logits = activation(logits)

    labels_np, logits_np = labels.numpy().astype('int'), \
        np.greater_equal(logits.numpy(), 0.5).astype('int')

    return metrics.accuracy_score(labels_np, logits_np)


def train_one_step(
        optimizer,
        text_encoder, meta_model, counts_model, multimode_model,
        statements_tf, justifications_tf, meta_tf, counts_tf, _labels,
        train_text_encoder=False,
        binary_classification=True):

    if train_text_encoder is False:
        statements_features = text_encoder(statements_tf)
        justifications_features = text_encoder(justifications_tf)

    with tf.GradientTape() as tape:
        if train_text_encoder:
            statements_features = text_encoder(statements_tf)
            justifications_features = text_encoder(justifications_tf)

        # print(meta_tf.shape, counts_tf.shape)
        meta_features = tf.dtypes.cast(meta_model(meta_tf), tf.float32)
        counts_features = tf.dtypes.cast(counts_model(counts_tf), tf.float32)

        multimode_feats = tf.concat([statements_features,
                                     justifications_features,
                                     meta_features,
                                     counts_features], -1)
        logits = multimode_model(multimode_feats)

        loss = compute_loss(logits=logits,
                            labels=_labels,
                            binary_classification=binary_classification)

    trainable_params = tape.watched_variables()
    grads = tape.gradient(loss, trainable_params)
    # update to weights
    optimizer.apply_gradients(zip(grads, trainable_params))

    accuracy = compute_accuracy(logits=logits,
                                labels=_labels,
                                binary_classification=binary_classification)

    # loss and accuracy is scalar tensor
    # return loss, weighted_f1, avg_hamming_loss, \
    #     avg_class_accuracies, exact_match_score
    return loss, accuracy


def predict_one_step(
        text_encoder, meta_model, counts_model, multimode_model,
        statements_tf, justifications_tf, meta_tf, counts_tf, _labels,
        binary_classification=True):

    is_training = False
    statements_features = text_encoder(statements_tf)
    justifications_features = text_encoder(justifications_tf)

    meta_features = tf.dtypes.cast(meta_model(meta_tf, is_training), tf.float32)
    counts_features = tf.dtypes.cast(
        counts_model(counts_tf, is_training), tf.float32)

    multimode_feats = tf.concat([statements_features,
                                 justifications_features,
                                 meta_features,
                                 counts_features], -1)
    logits = multimode_model(multimode_feats, is_training)

    loss = compute_loss(logits=logits,
                        labels=_labels,
                        binary_classification=binary_classification)

    accuracy = compute_accuracy(logits=logits,
                                labels=_labels,
                                binary_classification=binary_classification)

    return loss, accuracy


def train_one_epoch(df, batch_size, optimizer,
                    text_encoder, meta_model, counts_model,
                    multimode_model,
                    train_text_encoder=False,
                    repeat_first_batch=False, binary_classification=True):
    """Trains(feedforward and backward) the model for one step and returns
    useful metrics

    Args:
        dataset (utils.Dataset): Dataset object
        epoch (int): Current epoch number
        model (tf.keras.Model): Model object
        optimizer (tf.keras.optimizers): Description

    Returns:
        TYPE: multiclass mutlilabel metrics

    Raises:
        SfirstIteration: When the Dataset batch iterator is exhausted
    """
    train_ds = utils.batch(
        df, batch_size, repeat_first_batch=repeat_first_batch)
    # step = 0
    losses = []
    # weighted_f1s = []
    accuracies = []
    # # hamming_lossees = []
    # precisions = []
    # recalls = []
    # exact_match_scores = []
    try:
        while True:
            statements_tf, justifications_tf, meta_tf, counts_tf, \
                binary_labels_tf, labels_tf = next(train_ds)

            if binary_classification:
                _labels = binary_labels_tf
            else:
                _labels = labels_tf

            loss, accuracy = train_one_step(
                optimizer,
                text_encoder, meta_model, counts_model, multimode_model,
                statements_tf, justifications_tf, meta_tf, counts_tf, _labels,
                train_text_encoder,
                binary_classification)

            losses.append(loss.numpy())
            accuracies.append(accuracy)

    except StopIteration:
        return np.mean(losses), np.mean(accuracy)


def predict_one_epoch(df, batch_size, text_encoder, meta_model, counts_model,
                      multimode_model,
                      binary_classification=True):
    """Trains(feedforward and backward) the model for one step and returns
    useful metrics

    Args:
        dataset (utils.Dataset): Dataset object
        epoch (int): Current epoch number
        model (tf.keras.Model): Model object
        optimizer (tf.keras.optimizers): Description

    Returns:
        TYPE: multiclass mutlilabel metrics

    Raises:
        SfirstIteration: When the Dataset batch iterator is exhausted
    """
    test_ds = utils.batch(df, batch_size)
    # step = 0
    losses = []
    # weighted_f1s = []
    accuracies = []
    # # hamming_lossees = []
    # precisions = []
    # recalls = []
    # exact_match_scores = []
    try:
        while True:
            statements_tf, justifications_tf, meta_tf, counts_tf, \
                binary_labels_tf, labels_tf = next(test_ds)

            if binary_classification:
                _labels = binary_labels_tf
            else:
                _labels = labels_tf

            loss, accuracy = predict_one_step(
                text_encoder, meta_model, counts_model, multimode_model,
                statements_tf, justifications_tf, meta_tf, counts_tf, _labels,
                binary_classification)

            losses.append(loss.numpy())
            accuracies.append(accuracy)

    except StopIteration:
        return np.mean(losses), np.mean(accuracy)
