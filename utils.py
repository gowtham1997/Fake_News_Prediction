from keras.utils import to_categorical
import numpy as np


def get_suffixed_columns(df, suffix):
    return [col for col in df.columns if col.endswith(suffix)]


def encode_meta_features(df, meta_columns):
    df['subject_list_id'] = df['subject_list_id'].map(
        lambda x: np.array([int(i) for i in x.split(',')]))
    encoded_subject_id = np.array(df['subject_list_id'].to_list())
    # print(encoded_subject_id.shape)
    encoded_meta_cols = []
    for col in meta_columns:
        if col == 'subject_list_id':
            continue
        encoded_meta_col = to_categorical(df[col], df[col].max() + 1)
        # print(encoded_meta_col.shape)
        encoded_meta_cols.append(encoded_meta_col)

    return np.hstack((encoded_subject_id, ) + tuple(encoded_meta_col))
