# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 13:24:05 2020

@author: NGWASHIRONALD
"""
#This is a binary classification algorithm that predicts whether 
#1 the formation will produce sand or 0 the formation will not produce sand 
#PART 1: Data Preprocessing 

#Step 1: Importing the libraries 
import numpy as np
import matplotlib as plt
import pandas as pd 

#importing the dataset 
dataset = pd.read_csv("ANNDatasetMod.csv")
#Selecting the independent variables which will influence sanding 
#These variables include reservoir depth,overburden, pore pressure,
#Min and Max Horizontal stress, Poisson's ratio, Young's Modulus,
#Friction angle, and shale content.  
x = dataset.iloc[:,1:13].values
y = dataset.iloc[:,13].values
#There are no variables to encode so we move to splitting the dataset 
#from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#onehotencoder = OneHotEncoder(categorical_features = [1])
#x = onehotencoder.fit_transform(x).toarray()
#Splitting the dataset into the training set and Test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.65, random_state =109)
#feature Scaling to ease calculations and prevent one independent variable from dominating another
from  sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train =sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#PART 2: Building the ANN 

#Import Keras Libraries and Packages 
import keras
from keras.models import Sequential 
from keras.layers import Dense 

#Initializing the ANN
classifier = Sequential()

#Adding the input layer and the first hidden layer 
classifier.add(Dense(7, kernel_initializer = 'uniform', activation = 'relu', input_dim = 12))

#Adding the second hidden layer 
classifier.add(Dense(7, kernel_initializer = 'uniform', activation = 'relu'))

#Adding the output layer
classifier.add(Dense(1, kernel_initializer = 'uniform', activation = 'sigmoid'))

#Compiliing the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Fitting the ANN to the training set
classifier.fit(x_train, y_train,batch_size = 10, epochs = 200)

#PART 3: Making Predictions and Evaluating the Model

#Predicting the test set results 
y_pred = classifier.predict(x_test)
y_pred = (y_pred > 0.5)

#Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

