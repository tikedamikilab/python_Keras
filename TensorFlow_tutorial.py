import tensorflow_datasets as tfds
import tensorflow as tf

import matplotlib.pyplot as plt

def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_'+metric], '')
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_'+metric])
    plt.show()


dataset, info = tfds.load('imdb_reviews/subwords8k', with_info=True,as_supervised=True)
train_examples, test_examples = dataset['train'], dataset['test']

encoder = info.features['text'].encoder

print('Vocabulary size: {}'.format(encoder.vocab_size))

BUFFER_SIZE = 10000
BATCH_SIZE = 64

train_dataset = (train_examples
                .shuffle(BUFFER_SIZE)
                .padded_batch(BATCH_SIZE))

test_dataset = (test_examples
                .padded_batch(BATCH_SIZE))

