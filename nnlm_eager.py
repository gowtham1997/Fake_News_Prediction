import tensorflow_hub as hub
import tensorflow as tf
tf.enable_eager_execution()


class NNLMEncoder:
    # "https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1"
    def __init__(self, module_url, output_feature_size, name='tf_hub_encoder'):
        print(f'Initialising TensorFlow Hub Encoder op\n'
              f'This may take a long time on its first run, '
              f'as a pre-trained network module ({module_url}) needs to be '
              f'downloaded...')
        self._embed = hub.load(module_url)
        print('...Encoder op initialised')
        self._name = name
        self._output_feature_size = output_feature_size

    def __call__(self, features: tf.Tensor):

        embeddings = self._embed(features)
        return embeddings
