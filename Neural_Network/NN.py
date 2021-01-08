
import tensorflow as tf
import tensorflow_datasets as tfds
tfds.disable_progress_bar()
import math
import numpy as np
import matplotlib.pyplot as plt
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
import os
from tensorflow.keras.layers import Conv2D


#print(tf.version.VERSION)

dataset, metadata = tfds.load('fashion_mnist', as_supervised=True, with_info=True)
train_dataset, test_dataset = dataset['train'], dataset['test']


num_train_examples = metadata.splits['train'].num_examples
num_test_examples = metadata.splits['test'].num_examples
print("Number of training examples: {}".format(num_train_examples))
print("Number of test examples:     {}".format(num_test_examples))


def normalize(images, labels):
    images = tf.cast(images, tf.float32)
    images /= 255
    return images, labels




# The map function applies the normalize function to each element in the train
# and test datasets
train_dataset =  train_dataset.map(normalize)
test_dataset  =  test_dataset.map(normalize)

# The first time you use the dataset, the images will be loaded from disk
# Caching will keep them in memory, making training faster
train_dataset =  train_dataset.cache()
test_dataset  =  test_dataset.cache()

'''
Build the model
Building the neural network requires configuring the layers of the model, then compiling the model.

Setup the layers
The basic building block of a neural network is the layer. A layer extracts a representation from the data fed into it. Hopefully, a series of connected layers results in a representation that is meaningful for the problem at hand.

Much of deep learning consists of chaining together simple layers. Most layers, like tf.keras.layers.Dense, have internal parameters which are adjusted ("learned") during training.
'''


# Initialize model constructor
model = Sequential()


'''
TO DO
# Add layers sequentially
Keras tutorial is helpful: https://www.tensorflow.org/guide/keras/sequential_model
The score you get depends on your testing data accuracy.

0.00 <= test_accuracy < 0.85: 0/18
0.85 <= test_accuracy < 0.86: 3/18
0.86 <= test_accuracy < 0.87: 6/18
0.87 <= test_accuracy < 0.88: 9/18
0.88 <= test_accuracy < 0.89: 12/18
0.89 <= test_accuracy < 0.90: 15/18
0.90 <= test_accuracy < 1.00: 18/18

'''

################# TO DO ###################
##reference: https://www.kaggle.com/saraei1pyr/classifying-images-of-clothing
##reference: https://www.programcreek.com/python/example/89658/keras.layers.Conv2D
##Using reference help from kaggle for the layer adding and analyzation of the moddel

model.add(Conv2D(28, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(units=20, activation='softmax'))

###########################################

'''
Compile the model
Before the model is ready for training, it needs a few more settings. These are added during the model's compile step:

Loss function — An algorithm for measuring how far the model's outputs are from the desired output. The goal of training is this measures loss.
Optimizer —An algorithm for adjusting the inner parameters of the model in order to minimize loss.
Metrics —Used to monitor the training and testing steps. The following example uses accuracy, the fraction of the images that are correctly classified.
'''

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

'''
### Training is performed by calling the `model.fit` method:
1. Feed the training data to the model using `train_dataset`.
2. The model learns to associate images and labels.
3. The `epochs=5` parameter limits training to 5 full iterations of the training dataset, so a total of 5 * 60000 = 300000 examples.

(Don't worry about `steps_per_epoch`, the requirement to have this flag will soon be removed.)
'''

BATCH_SIZE = 32
train_dataset = train_dataset.cache().repeat().shuffle(num_train_examples).batch(BATCH_SIZE)
test_dataset = test_dataset.cache().batch(BATCH_SIZE)


'''
TO DO
# Decide how mnay epochs you want
'''

model.fit(train_dataset, epochs=20, steps_per_epoch=math.ceil(num_train_examples/BATCH_SIZE))


'''
As the model trains, the loss and accuracy metrics are displayed. This model reaches an accuracy of about 0.88 (or 88%) on the training data.
'''
model.save('my_model')


test_loss, test_accuracy = model.evaluate(test_dataset, steps=math.ceil(num_test_examples/32))
print('Accuracy on test dataset:', test_accuracy)
