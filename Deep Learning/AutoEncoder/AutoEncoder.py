# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow
from keras.layers import (Dense, Convolution2D, Activation, MaxPooling2D, Flatten, Reshape
, Input, UpSampling2D, ZeroPadding2D)
from keras.models import Sequential, Model, load_model
from keras.utils import np_utils

x = pd.read_csv('train.csv')
X_ = np.array(x)

X = X_[:, 1:]
X = X/255.0
X = X.reshape((X.shape[0], 28, 28, 1))
y = X_[:, 0]

# Encoding pic to 64
inp = Input(shape=(28, 28, 1))
c1 = Convolution2D(32, (3, 3), activation='relu', border_mode='valid')(inp)
c2 = Convolution2D(16, (3, 3), activation='relu', border_mode='valid')(c1)
m1 = MaxPooling2D((2,2))(c2)
c3 = Convolution2D(8, (3, 3), activation='relu', border_mode='same')(m1)
f1 = Flatten()(c3)
fc1 = Dense(64, activation='relu')(f1)

# Decoder
fc2 = Dense(800, activation='relu')(fc1)
re2 = Reshape((10, 10, 8))(fc2)
zp1 = ZeroPadding2D((1,1))(re2)
c4 = Convolution2D(16, (3, 3), activation='relu', border_mode='same')(zp1)
u1 = UpSampling2D((2, 2))(c4)
zp2 = ZeroPadding2D((1, 1))(u1)
c5 = Convolution2D(32, (3, 3), activation='relu', border_mode='same')(zp2)
zp3 = ZeroPadding2D((1, 1))(c5)
c6 = Convolution2D(1, (3, 3), activation='relu', border_mode='same')(zp3)

model = Model(input=inp, output=c6)
encoder = Model(input=inp, output=fc1)

model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

hist = model.fit(X, X, 
                 epochs=15,
                 shuffle=True,
                 batch_size=256,
                 validation_split=0.20)

plt.figure(0)
plt.plot(hist.history['loss'], 'g')
plt.plot(hist.history['val_loss'], 'r')
plt.show()

encoder.save('enc_64d.h5')

X_in = X[:10, :]
Y_out = model.predict(X_in)

X_in = X_in.reshape((X_in.shape[0], 28, 28))
Y_out = Y_out.reshape((Y_out.shape[0], 28, 28))

for ix in range(1, 10):
    print(20*'#')
    plt.figure(ix)
    plt.imshow(X_in[ix], cmap='gray')
    plt.figure(3*ix)
    plt.imshow(Y_out[ix], cmap='gray')
    plt.show()
    print(20*'#')



