# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 20:25:17 2021

@author: NGWASHIRONALD
"""

#This is a binary classification algorithm that predicts whether 
#1 the formation will produce sand or 0 the formation will not produce sand 
#PART 1: Data Preprocessing 

#Step 1: Importing the libraries 
import numpy as np
import matplotlib.pyplot as mtp  
import pandas as pd 
from scipy import stats
import seaborn as sns; sns.set()

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
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.30, random_state =109)
#feature Scaling to ease calculations and prevent one independent variable from dominating another
from  sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train =sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#Import svm model
from sklearn import svm

#Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets
clf.fit(x_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(x_test)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
#Creating the Confusion matrix  
from sklearn.metrics import confusion_matrix  
cm= confusion_matrix(y_test, y_pred)  
print(cm)
# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:",metrics.precision_score(y_test, y_pred))

# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(y_test, y_pred))

from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))




from matplotlib.colors import ListedColormap  
x_set, y_set = x_train, y_train  
x1, x2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step  =0.01),  
np.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01)) 
xpred = np.array([x1.ravel(), x2.ravel()] + [np.repeat(0, x1.ravel().size) for _ in range(10)]).T
pred =  clf.predict(xpred).reshape(x1.shape)
mtp.contour(x1,x2,pred,alpha = 0.75, cmap = ListedColormap(('red','green')))

mtp.xlim(x1.min(), x1.max())  
mtp.ylim(x2.min(), x2.max())  
for i, j in enumerate(np.unique(y_set)):  
    mtp.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],  
        c = ListedColormap(('red', 'green'))(i), label = j)  
mtp.title('SVM classifier (Training set)')  
mtp.xlabel('Age')  
mtp.ylabel('Estimated Salary')  
mtp.legend()  
mtp.show()  

x_set, y_set = x_test, y_test  
x1, x2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step  =0.01),  
np.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01)) 
xpred = np.array([x1.ravel(), x2.ravel()] + [np.repeat(0, x1.ravel().size) for _ in range(10)]).T
pred =  clf.predict(xpred).reshape(x1.shape)
mtp.contour(x1,x2,pred,alpha = 0.75, cmap = ListedColormap(('red','green')))

mtp.xlim(x1.min(), x1.max())  
mtp.ylim(x2.min(), x2.max())  
for i, j in enumerate(np.unique(y_set)):  
    mtp.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],  
        c = ListedColormap(('red', 'green'))(i), label = j)  
mtp.title('SVM classifier (Training set)')  
mtp.xlabel('Age')  
mtp.ylabel('Estimated Salary')  
mtp.legend()  
mtp.show() 
