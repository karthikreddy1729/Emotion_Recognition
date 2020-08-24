# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 23:13:41 2020

@author: karth
"""

import sys, os
import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils import np_utils

df=pd.read_csv('dataset/fer2013.csv')

X_train=[]
y_train=[]
X_test=[]
y_test=[]

for index, row in df.iterrows():
    val=row['pixels'].split(" ")
    if 'Training' in row['Usage']:
        X_train.append(np.array(val,'float32'))
        y_train.append(row['emotion'])
    elif 'PublicTest' in row['Usage']:
        X_test.append(np.array(val,'float32'))
        y_test.append(row['emotion'])


features = 64
labels = 7
batch_size = 64
epochs = 30
width = 48
height = 48


X_train = np.array(X_train,'float32')
y_train = np.array(y_train,'float32')
X_test = np.array(X_test,'float32')
y_test = np.array(y_test,'float32')

y_train=np_utils.to_categorical(y_train, num_classes=labels)
y_test=np_utils.to_categorical(y_test, num_classes=labels)

#normalizing data between oand 1
X_train -= np.mean(X_train, axis=0)
X_train /= np.std(X_train, axis=0)

X_test -= np.mean(X_test, axis=0)
X_test /= np.std(X_test, axis=0)

X_train = X_train.reshape(X_train.shape[0], 48, 48, 1)

X_test = X_test.reshape(X_test.shape[0], 48, 48, 1)

#designing the cnn
model = Sequential()

#1st convolution layer
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(X_train.shape[1:])))
model.add(Conv2D(64,kernel_size= (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
model.add(Dropout(0.5))

#2nd convolution layer
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
model.add(Dropout(0.5))

#3rd convolution layer
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))

model.add(Flatten())

#fully connected neural networks
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(labels, activation='softmax'))


#Compliling the model
model.compile(loss=categorical_crossentropy,
              optimizer=Adam(),
              metrics=['accuracy'])

#Training the model
model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_test, y_test),
          shuffle=True)


fer_json = model.to_json()
with open("models/fer.json", "w") as json_file:
    json_file.write(fer_json)
model.save_weights("models/model.h5")





