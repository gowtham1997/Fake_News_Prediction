from keras.utils import to_categorical
import numpy as np

import tensorflow as tf
tf.enable_eager_execution()


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
