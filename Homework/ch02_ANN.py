# https://www.tensorflow.org/datasets/keras_example
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model
from tensorflow.keras import models
import tensorflow_datasets as tfds
from sklearn.metrics import classification_report

# ! pip install tensorflow_datasets

# Setting hyperparameters
OUTPUT_PATH = 'denseModel1.png'
BS          = 40   #batch_size
NUM_EPOCHS  = 10

# I.Preprocessing

# 1.load MNIST Dataset
mnist = tf.keras.datasets.mnist
ds, info = tfds.load('mnist', split='train', with_info=True)

# 2.Splitting into train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_val  = x_train[55000:60000]
x_train = x_train[0:55000]

y_val  = y_train[55000:60000]
y_train = y_train[0:55000]

x_train = x_train.reshape(55000, 784)
x_val = x_val.reshape(5000, 784)
x_test = x_test.reshape(10000, 784)

# 3.Normalize data
x_train = x_train.astype('float32')
x_val = x_val.astype('float32')
x_test = x_test.astype('float32')

gray_scale = 255
x_train /= gray_scale
x_val /= gray_scale
x_test /= gray_scale

# 4.label to one hot encoding value
num_classes = 10
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_val = tf.keras.utils.to_categorical(y_val, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

# II.Define Model
model1 = Sequential() 
model1.add( Dense (256, activation = 'relu', input_shape= x_train[0].shape))
model1.add( Dense (128, activation = 'relu'))
model1.add( Dense (128, activation = 'relu'))
model1.add( Dense (128, activation = 'relu'))
model1.add( Dense (128, activation = 'relu'))
model1.add( Dense ( 64, activation = 'relu'))
model1.add( Dense ( 64, activation = 'relu'))
model1.add( Dense ( 64, activation = 'relu'))
model1.add( Dense ( 64, activation = 'relu'))
model1.add( Dense ( 10, activation = 'softmax'))
print(model1.summary())
plot_model(model1, to_file="Model1.png", show_shapes=True)

# III.Train Model
model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
H = model1.fit(x_train, y_train, batch_size = BS, epochs= NUM_EPOCHS, validation_split = 0.2)

# IV.Evaluate Model
print("[INFO] evaluating network...")
print(model1.evaluate(x_test, y_test))

# V.Use Model
print("[INFO] predicting classes...")
predIdxs = model1.predict(x_test)
predIdxs = np.argmax(predIdxs, axis=1)
y_test1  = np.argmax(y_test, axis=1)
print(classification_report(y_test1, predIdxs, target_names=info.features["label"].names))

# VI.Save the model
# (10) Using Keras to save the model structure and weights
model1.save('./keras.model1.h5')

del model1
model1 = models.load_model('./keras.model1.h5')
print("[INFO] evaluating network after loading the model...")
print(model1.evaluate(x_test, y_test))

