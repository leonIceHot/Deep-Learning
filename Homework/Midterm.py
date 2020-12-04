# Getting Started with TensorFlow 2
# https://www.kdnuggets.com/2020/07/getting-started-tensorflow2.html

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import preprocessing
from tensorflow.keras import models

OUTPUT_PATH = 'denseModel1.png'
BS          = 40
NUM_EPOCHS  = 800

# 一、準備數據
# (1) Load Iris flower dataset
wine = load_wine()

# Show the first five data
X = pd.DataFrame(data = wine.data, columns = wine.feature_names)
print(X.head())

y = pd.DataFrame(data = wine.target, columns = ['wineType'])
print('\n')
print(y.head())
print(y.wineType.value_counts())

# Show the iris class names
print('\n')
print(wine.target_names)

# (2) Splitting into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# (3) Normalization of data
print('\n')
print(X_train.var())
print(X_test.var())
# Convert X_train and X_test from DataFrame to float64
scalar = preprocessing.StandardScaler().fit(X_train)
X_train = scalar.transform(X_train)
X_test  = scalar.transform(X_test)
print('\n')

# (4) Categorical data into one-hot vector
y_train = tf.keras.utils.to_categorical(y_train)
y_test  = tf.keras.utils.to_categorical(y_test)

# 二、定義模型
# (5) Machine learning model
model1 = Sequential() 
model1.add( Dense (32, activation = 'relu', input_shape= X_train[0].shape))
model1.add( Dense (64, activation = 'relu'))
model1.add( Dense (32, activation = 'relu'))
model1.add( Dense (3, activation = 'softmax'))
print(model1.summary())
plot_model(model1, to_file="Model1.png", show_shapes=True)

# 三、訓練模型
# (6) Settings of optimizer and loss function
model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
H = model1.fit(X_train, y_train, batch_size = BS, epochs= NUM_EPOCHS, validation_split = 0.1)

'''
# (7) Plot of acc and val_acc vs. epochs
N = NUM_EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,N), H.history['loss'], label="train_loss")
plt.plot(np.arange(0,N), H.history['val_loss'], label="val_loss")
plt.plot(np.arange(0,N), H.history['acc'], label="train_acc")
plt.plot(np.arange(0,N), H.history['val_acc'], label="val_acc")
plt.title("Training Loss vs. Accuracy without Regularization")
plt.xlabel('Epoch #')
plt.ylabel('Loss/Accuracy')
plt.legend(loc='upper right')
plt.savefig(OUTPUT_PATH)
'''

# (7) Plot of acc and val_acc vs. epochs
N = NUM_EPOCHS
plt.style.use("classic")
fig, ax1 = plt.subplots()
ax1.plot(np.arange(0,N), H.history['acc'], c="b", label="train_acc")
ax1.plot(np.arange(0,N), H.history['val_acc'], c="g", label="val_acc")
plt.title('Training Loss vs. Accuracy without Regularization')
ax1.set_ylabel('Accuracy')
ax1.set_xlabel('Epoch')
ax1.set_ylim(0.0, 1.1)
plt.legend(loc='center left', shadow=True)

ax2 = ax1.twinx()

ax2.plot(np.arange(0,N), H.history['loss'], c="r", label="train_loss")
ax2.plot(np.arange(0,N), H.history['val_loss'], c="c", label="val_loss")
ax2.set_ylabel('Loss')
ax2.set_xlabel('Epoch')
plt.legend(loc='center right', shadow=True)
plt.savefig(OUTPUT_PATH)

# 四、評估模型
# (8) Evaluate the test dataset
print("[INFO] evaluating network...")
print(model1.evaluate(X_test, y_test))

# 五、使用模型
# (9) Predict the test dataset
print("[INFO] predicting classes...")
predIdxs = model1.predict(X_test)
predIdxs = np.argmax(predIdxs, axis=1)
y_test1  = np.argmax(y_test, axis=1)
print(classification_report(y_test1, predIdxs, target_names=wine.target_names))

# 六、保存模型
# (10) Using Keras to save the model structure and weights
model1.save('./keras.model1.h5')

del model1
model1 = models.load_model('./keras.model1.h5')
print("[INFO] evaluating network after loading the model...")
print(model1.evaluate(X_test, y_test))


 

