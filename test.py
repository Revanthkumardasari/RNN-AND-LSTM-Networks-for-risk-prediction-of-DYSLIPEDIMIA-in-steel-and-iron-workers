import pandas as pd
import numpy as np
from Keras.models import Sequential
from keras.layers.core import Dense,Activation,Dropout, Flatten
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import keras.layers

le = LabelEncoder()

dataset = pd.read_csv('dataset.csv')
dataset['Fibrosis stage'] = pd.Series(le.fit_transform(dataset['Fibrosis stage']))
temp = dataset.values
Y = temp[:,28]
Y = Y.astype('int')                      
dataset.drop(['Hospital','Metabolic Syndrome'], axis = 1,inplace=True)
dataset.fillna(0, inplace = True)

dataset = dataset.values
X = dataset[:,0:dataset.shape[1]] 
print(Y)
print(X)
print(Y.shape)
print(X.shape)

Y1 = to_categorical(Y)
#train_x, test_x, train_y, test_y = train_test_split(XX, Y1, test_size=0.2)
'''
rnn = Sequential()
rnn.add(Dense(256, input_dim=X.shape[1], activation='relu', kernel_initializer = "uniform"))
rnn.add(Dense(128, activation='relu', kernel_initializer = "uniform"))
rnn.add(Dense(2, activation='softmax',kernel_initializer = "uniform"))
rnn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(rnn.summary())
rnn_history = rnn.fit(X, Y1, epochs=10, batch_size=8)
rnn_history = rnn_history.history
rnn_history = rnn_history['accuracy']
acc = rnn_history[8] * 100
print(acc)
'''

XX = X.reshape((X.shape[0], X.shape[1], 1)) 
model = Sequential()
model.add(keras.layers.LSTM(512,input_shape=(X.shape[1], 1)))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
rnn_history = model.fit(XX, Y1, epochs=50, batch_size=16)
rnn_history = rnn_history.history
rnn_history = rnn_history['accuracy']
acc = rnn_history[8] * 100
print(acc)












