import warnings
import sklearn.exceptions as exp
# from tqdm import tqdm
import typing as typ
import os
# to disable tf warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf
tf.enable_eager_execution()
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, \
    MaxPool1D, Conv1D, Embedding
# ignore sklearn's warnings
warnings.filterwarnings("ignore", category=exp.UndefinedMetricWarning)
flags = tf.app.flags
FLAGS = flags.FLAGS

# keeping batchnorm momentum less(than default 0.99) makes sure the val loss is
# stable throughout training
# NOTE: Also while training, the batchnorm layer placed inside Keras sequential
# gives an index(out of range error) while placing it outside works
BATCH_NORM_MOMENTUM = 0.9


class MetaModel(tf.keras.Model):

    """Model object

    Attributes:
        model (TYPE): Description
        model (tf.Keras.Model)
    """

    def __init__(self, input_shape: typ.Tuple[int, int, int],
                 fc_units=1024):
        """initialise the model object

        Args:
            input_shape (typ.Tuple[int, int, int]): input_shape
            num_classes (int): number of classes
        """
        super(MetaModel, self).__init__()
        # self.model = tf.keras.Sequential([
        #     tf.keras.layers.Dense(units=1024, input_shape=input_shape),
        #     tf.keras.layers.Dense(units=num_classes),
        #     # tf.keras.layers.BatchNormalization(momentum=BATCH_NORM_MOMENTUM)
        # ])
        self.d1 = tf.keras.layers.Dense(units=fc_units, input_shape=input_shape)
        self.d2 = tf.keras.layers.Dense(
            units=fc_units, input_shape=(None, fc_units))

        self.bn1 = tf.keras.layers.BatchNormalization(
            momentum=BATCH_NORM_MOMENTUM)
        self.bn2 = tf.keras.layers.BatchNormalization(
            momentum=BATCH_NORM_MOMENTUM)

    def __call__(self, _input, is_training=True):
        """Run the model.

        Args:
            _input (TYPE): input to the model, mostly the embeddings

        Returns:
            TYPE: Outputs after passing through the dense layers
        """
        # return self.bn1(self.model(_input), training=is_training)
        x = tf.nn.relu(self.bn1(self.d1(_input), training=is_training))
        x = tf.nn.relu(self.bn2(self.d2(x), training=is_training))
        return x


class MultiModeModel(tf.keras.Model):

    """Model object

    Attributes:
        model (TYPE): Description
        model (tf.Keras.Model)
    """

    def __init__(self, input_shape: typ.Tuple[int, int, int], fc_units=1024,
                 num_classes=6):
        """initialise the model object

        Args:
            input_shape (typ.Tuple[int, int, int]): input_shape
            num_classes (int): number of classes
        """
        super(MultiModeModel, self).__init__()
        # self.model = tf.keras.Sequential([
        #     tf.keras.layers.Dense(units=1024, input_shape=input_shape),
        #     tf.keras.layers.Dense(units=num_classes),
        #     # tf.keras.layers.BatchNormalization(momentum=BATCH_NORM_MOMENTUM)
        # ])
        self.d1 = tf.keras.layers.Dense(
            units=1024, input_shape=input_shape)
        self.d2 = tf.keras.layers.Dense(
            units=num_classes, input_shape=(None, fc_units))

        self.bn1 = tf.keras.layers.BatchNormalization(
            momentum=BATCH_NORM_MOMENTUM)
        # self.bn2 = tf.keras.layers.BatchNormalization(
        #     momentum=BATCH_NORM_MOMENTUM)

    def __call__(self, _input, is_training=True):
        """Run the model.

        Args:
            _input (TYPE): input to the model, mostly the embeddings

        Returns:
            TYPE: Outputs after passing through the dense layers
        """
        # return self.bn1(self.model(_input), training=is_training)
        # x = tf.nn.relu(self.bn1(self.d1(_input), training=is_training))
        x = tf.nn.relu(self.bn1(self.d1(_input), training=is_training))
        x = self.d2(x)

        return x


class CountModel(tf.keras.Model):

    """Model object

    Attributes:
        model (TYPE): Description
        model (tf.Keras.Model)
    """

    def __init__(self, input_shape: typ.Tuple[int, int, int],
                 fc_units=1024):
        """initialise the model object

        Args:
            input_shape (typ.Tuple[int, int, int]): input_shape
            num_classes (int): number of classes
        """
        super(CountModel, self).__init__()
        # self.model = tf.keras.Sequential([
        #     tf.keras.layers.Dense(units=1024, input_shape=input_shape),
        #     tf.keras.layers.Dense(units=num_classes),
        #     # tf.keras.layers.BatchNormalization(momentum=BATCH_NORM_MOMENTUM)
        # ])
        self.d1 = tf.keras.layers.Dense(units=fc_units, input_shape=input_shape)
        self.d2 = tf.keras.layers.Dense(
            units=fc_units, input_shape=(None, fc_units))

        self.bn1 = tf.keras.layers.BatchNormalization(
            momentum=BATCH_NORM_MOMENTUM)
        self.bn2 = tf.keras.layers.BatchNormalization(
            momentum=BATCH_NORM_MOMENTUM)

    def __call__(self, _input, is_training=True):
        """Run the model.

        Args:
            _input (TYPE): input to the model, mostly the embeddings

        Returns:
            TYPE: Outputs after passing through the dense layers
        """
        # return self.bn1(self.model(_input), training=is_training)
        x = tf.nn.relu(self.bn1(self.d1(_input), training=is_training))
        x = tf.nn.relu(self.bn2(self.d2(x), training=is_training))
        return x


class BiLSTMDense(tf.keras.Model):

    """Model object

    Attributes:
        model (TYPE): Description
        model (tf.Keras.Model)
    """

    def __init__(self, vocab_size, emb_dim, embedding_matrix, seq_len):
        """initialise the model object

        Args:
            input_shape (typ.Tuple[int, int, int]): input_shape
            num_classes (int): number of classes
        """
        super(BiLSTMDense, self).__init__()
        self.model = tf.keras.Sequential()
        self.model.add(Embedding(vocab_size, emb_dim, embeddings_initializer=[
                       embedding_matrix],
            input_length=seq_len, trainable=False))
        self.model.add(Bidirectional(LSTM(50)))
        # model.add(Bidirectional(LSTM(10)))
        self.model.add(Dense(128))
        self.model.add(tf.nn.relu)

    def __call__(self, _input):
        """Run the model.

        Args:
            _input (TYPE): input to the model, mostly the embeddings

        Returns:
            TYPE: Outputs after passing through the dense layers
        """
        return self.model(_input)


class Conv1DModel(tf.keras.Model):

    """Model object

    Attributes:
        model (TYPE): Description
        model (tf.Keras.Model)
    """

    def __init__(self, vocab_size, emb_dim, embedding_matrix, seq_len):
        """initialise the model object

        Args:
            input_shape (typ.Tuple[int, int, int]): input_shape
            num_classes (int): number of classes
        """
        super(Conv1DModel, self).__init__()
        self.model = tf.keras.Sequential()
        self.model.add(Embedding(vocab_size, emb_dim, embeddings_initializer=[
                       embedding_matrix],
            input_length=seq_len, trainable=False))
        self.model.add(Conv1D(filters=64, kernel_size=(2, 3, 4),
                              activation='relu'))
        # model.add(Bidirectional(LSTM(10)))
        self.model.add(MaxPool1D(pool_size=2))

    def __call__(self, _input):
        """Run the model.

        Args:
            _input (TYPE): input to the model, mostly the embeddings

        Returns:
            TYPE: Outputs after passing through the dense layers
        """
        return self.model(_input)
