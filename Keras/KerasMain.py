# Sequential - pipeline
# core       - layers
# MNIST      - db (handrite numbers)

# model = Sequential()
# 1. type of modules ?

# model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(1,28,28)))
# 2. type of layers

# dropout method (layer) (remove some neurons randomly)
# 3. read about it

# model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
# 4.1 loss ?
# 4.2 optimizaer
# 4.3 metrics

# model.summary()
# 5. ?

# model.fit(X_train, Y_train, batch_size=32, nb_epoch=10, verbose=1)
# 6. batch size ? nb_epoch ?

# score = model.evaluate(X_test, Y_test, verbose=0)


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD,Adam
from sklearn.preprocessing import LabelEncoder
#os.chdir('/Users/akashsrivastava/Desktop/MachineLearning/kaggle/iris-keras')


from sklearn import datasets

# def run_model(X_train, X_test, y_train, y_test):
#     encoder = LabelEncoder()
#     y_train = encoder.fit_transform(y_train)
#     y_train = pd.get_dummies(y_train).values
#     y_test = encoder.fit_transform(y_test)
#     y_test = pd.get_dummies(y_test).values
#
#     model = Sequential()
#     model.add(Dense(10, input_shape=(4,), activation='tanh'))
#     model.add(Dense(8, activation='tanh'))
#     model.add(Dense(6, activation='tanh'))
#     model.add(Dense(3, activation='softmax'))
#
#     model.compile(Adam(lr=0.04), 'categorical_crossentropy', metrics=['accuracy'])
#     model.fit(X_train, y_train, epochs=10,verbose=False)
#     #y_pred = model.predict(X_test)
#     score, acc = model.evaluate(X_test, y_test)#, batch_size=batch_size)
#     return acc

samples = datasets.load_iris()
X = samples.data
y = samples.target
df = pd.DataFrame(data=X)
df.columns = samples.feature_names
df['Target'] = y

dataset = df #pd.read_csv('../input/Iris.csv')

# import seaborn as sns
# sns.set(style="ticks")
# sns.set_palette("husl")
# sns.pairplot(dataset.iloc[:,1:6],hue="Species")

#Splitting the data into training and test test
#X = dataset.iloc[:,1:5].values
#y = dataset.iloc[:,5].values


encoder =  LabelEncoder()
y1 = encoder.fit_transform(y)

Y = pd.get_dummies(y1).values


from sklearn.model_selection import train_test_split
X_train,X_test, y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=0)
#Defining the model




model = Sequential()

model.add(Dense(10,input_shape=(4,),activation='tanh'))
model.add(Dense(8,activation='tanh'))
model.add(Dense(6,activation='tanh'))
model.add(Dense(3,activation='softmax'))

model.compile(Adam(lr=0.04),'categorical_crossentropy',metrics=['accuracy'])

#model.summary()

#fitting the model and predicting
model.fit(X_train,y_train,epochs=100)
y_pred = model.predict(X_test)

y_test_class = np.argmax(y_test,axis=1)
y_pred_class = np.argmax(y_pred,axis=1)

#Accuracy of the predicted values
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test_class,y_pred_class))
print(confusion_matrix(y_test_class,y_pred_class))
print("Finished")