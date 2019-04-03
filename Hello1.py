import pandas as pd
import numpy as np
import theano
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.callbacks import TensorBoard

model = Sequential()
model.add(Dense(3, input_dim=2, init='uniform'))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(3, init='uniform'))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(3, init='uniform'))
model.add(Activation('softmax'))
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

tensorboard = TensorBoard(log_dir='C:/Users/Rubens/Anaconda3/envs/tensorflow/Scripts/plot_keras', histogram_freq=1,
                          write_graph=True, write_images=False)
tensorboard.set_model(model) 


aa=pd.read_csv('DadosTeseLogit3.csv',sep=',',header=0)

X_train=aa.iloc[:,[18,29]]
y_train=pd.get_dummies(aa.iloc[:,30])

model.fit(np.array(X_train), np.array(y_train),
          nb_epoch=20,
          batch_size=16)

'''RUN
tensorboard --logdir=C:/Users/Rubens/Anaconda3/envs/tensorflow/Scripts/plot_keras'''
