from tkinter import *
import tkinter
from tkinter import filedialog
import numpy as np
from tkinter.filedialog import askopenfilename
import pandas as pd 
from tkinter import simpledialog
import matplotlib.pyplot as plt
from PIL import ImageTk, Image
import os
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense,Activation,Dropout, Flatten
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import keras.layers
main=tkinter.Tk()
main.title("Research on Risk Prediction of Dyslipidemia in Steel and Iron workers Based on Recurrent Neural Network and LSTM Neural Network")

global filename
global rnn_acc,lstm_acc
global classifier
global X, Y, Y1
global dataset
global le

def upload():
    global filename
    global dataset
    filename = filedialog.askopenfilename(initialdir = "dataset")
    text.delete('1.0', END)
    text.insert(END,filename+' Loaded\n\n')
    dataset = pd.read_csv(filename)
    text.insert(END,str(dataset.head))

def preprocess():
    global dataset
    global X, Y, Y1
    global le
    text.delete('1.0', END)
    le = LabelEncoder()
    dataset['Fibrosis stage'] = pd.Series(le.fit_transform(dataset['Fibrosis stage']))
    temp = dataset.values
    Y = temp[:,28]
    Y = Y.astype('int')                      
    dataset.drop(['Hospital','Metabolic Syndrome'], axis = 1,inplace=True)
    dataset.fillna(0, inplace = True)
    dataset = dataset.values
    X = dataset[:,0:dataset.shape[1]] 
    print(Y)
    text.insert(END,str(X))
    print(Y.shape)
    print(X.shape)
    Y1 = to_categorical(Y)    
    
    text.insert(END,"\n\nTotal Records after preprocessing are : "+str(len(X))+"\n")

def runRNN():
    global classifier
    text.delete('1.0', END)
    global rnn_acc
    rnn = Sequential() #creating RNN model object
    rnn.add(Dense(256, input_dim=X.shape[1], activation='relu', kernel_initializer = "uniform")) #defining one layer with 256 filters to filter dataset
    rnn.add(Dense(128, activation='relu', kernel_initializer = "uniform"))#defining another layer to filter dataset with 128 layers
    rnn.add(Dense(2, activation='softmax',kernel_initializer = "uniform")) #after building model need to predict two classes such as normal or Dyslipidemia disease
    rnn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) #while filtering and training dataset need to display accuracy 
    print(rnn.summary()) #display rnn details
    rnn_acc = rnn.fit(X, Y1, epochs=60, batch_size=64) #start building RNN model
    values = rnn_acc.history #save each epoch accuracy and loss
    values = values['accuracy']
    acc = values[59] * 100
    text.insert(END,'RNN Prediction Accuracy : '+str(acc)+"\n\n")
    classifier = rnn

def runLSTM():
    global lstm_acc
    XX = X.reshape((X.shape[0], X.shape[1], 1)) 
    model = Sequential() #creating LSTM model object
    model.add(keras.layers.LSTM(512,input_shape=(X.shape[1], 1))) #defining LSTM layer in sequential object
    model.add(Dropout(0.5)) #removing irrelevant dataset features
    model.add(Dense(256, activation='relu'))#create another layer
    model.add(Dense(2, activation='softmax'))#predict two values as normal or Dyslipidemia disease
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])#calculate accuracy
    print(model.summary())
    lstm_acc = model.fit(XX, Y1, epochs=60, batch_size=64) #start training model
    values = lstm_acc.history
    values = values['accuracy']
    acc = values[59] * 100
    text.insert(END,'LSTM Prediction Accuracy : '+str(acc)+"\n\n")
        

def predict():
    text.delete('1.0', END)
    file = filedialog.askopenfilename(initialdir="dataset")
    test = pd.read_csv(file)
    test['Fibrosis stage'] = pd.Series(le.fit_transform(test['Fibrosis stage']))
    test.drop(['Hospital','Metabolic Syndrome'], axis = 1,inplace=True)
    test.fillna(0, inplace = True)
    test = test.values
    test = test[:,0:test.shape[1]] 
    y_pred = classifier.predict(test)
    for i in range(len(test)):
        predict = np.argmax(y_pred[i])
        print(str(predict))
        if predict == 0:
            text.insert(END,"X=%s, Predicted = %s" % (test[i], 'No Dyslipidemia disease detected')+"\n\n")
        else:
            text.insert(END,"X=%s, Predicted = %s" % (test[i], 'Dyslipidemia disease detected')+"\n\n")
    
def graph():
    lstm_temp = lstm_acc.history
    lstm_accuracy = lstm_temp['accuracy']
    lstm_loss = lstm_temp['loss']

    rnn_temp = rnn_acc.history
    rnn_accuracy = rnn_temp['accuracy']
    rnn_loss = rnn_temp['loss']
    
    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy/Loss')
    plt.plot(lstm_accuracy, 'ro-', color = 'red')
    plt.plot(rnn_accuracy, 'ro-', color = 'green')
    plt.plot(lstm_loss, 'ro-', color = 'blue')
    plt.plot(rnn_loss, 'ro-', color = 'orange')
    plt.legend(['LSTM Accuracy', 'RNN Accuracy','LSTM Loss','RNN Loss'], loc='upper left')
    #plt.xticks(wordloss.index)
    plt.title('LSTM & RNN Accuracy & Loss Graph')
    plt.show()

def close():
  main.destroy()
   
font = ('times', 15, 'bold')
title = Label(main, text='Research on Risk Prediction of Dyslipidemia in Steel Workers Based on Recurrent Neural Network and LSTM Neural Network', justify=LEFT)
title.config(bg='yellow', fg='blue')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=100,y=5)
title.pack()

font1 = ('times', 12, 'bold')
uploadButton = Button(main, text="Upload Dyslipidemia Dataset", command=upload)
uploadButton.place(x=10,y=100)
uploadButton.config(font=font1)  

preprocessButton = Button(main, text="Preprocess Dataset", command=preprocess)
preprocessButton.place(x=300,y=100)
preprocessButton.config(font=font1)

rnnButton = Button(main, text="Run RNN Algorithm", command=runRNN)
rnnButton.place(x=480,y=100)
rnnButton.config(font=font1)

lstmButton = Button(main, text="Run LSTM Algorithm", command=runLSTM)
lstmButton.place(x=670,y=100)
lstmButton.config(font=font1)

graphButton = Button(main, text="Accuracy Comparison Graph", command=graph)
graphButton.place(x=10,y=150)
graphButton.config(font=font1)

predictButton = Button(main, text="Predict Disease using Test Data", command=predict)
predictButton.place(x=300,y=150)
predictButton.config(font=font1)

closeButton = Button(main, text="Close Application", command=close)
closeButton.place(x=10,y=200)
closeButton.config(font=font1)

font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=160)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=250)
text.config(font=font1) 
main.config(bg="coral")
main.mainloop()
