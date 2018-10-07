import tensorflow
import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Dense, Activation
from keras.models import Sequential
from keras.utils import np_utils

x = pd.read_csv("./fashion-mnist_test.csv")
X_ = np.array(x)
X = X_[:,1:]
X = X/255.0
y = X_[:,0]

print(X.shape, y.shape)

y = np_utils.to_categorical(y)
print(y.shape, y[:10])

X_train = X[:8000, :]
Y_train = y[:8000, :]
X_val = X[8000:, :]
Y_val = y[8000:, :]

model = Sequential()
model.add(Dense(256, input_shape=(784,)))
model.add(Activation('relu'))

model.add(Dense(64))
model.add(Activation('relu'))

model.add(Dense(10))
model.add(Activation('sigmoid'))

print(model.summary())

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
hist = model.fit(X_train, Y_train, batch_size=256, epochs=50, verbose=2, validation_data=(X_val, Y_val))

plt.figure(0)
plt.plot(hist.history['loss'], 'r')
plt.plot(hist.history['val_loss'], 'black')
plt.plot(hist.history['acc'], 'g')
plt.plot(hist.history['val_acc'], 'b')
plt.show()