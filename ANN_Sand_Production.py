# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 14:42:24 2021

@author: NGWASHIRONALD
"""

# Import pandas 
import pandas as pd

#import numpy
import numpy as np

# Read in white wine data 
Sand = pd.read_csv("Sand.csv")

# Read in red wine data 
No_Sand = pd.read_csv("No_Sand.csv")

# Print info on Sand data
print(Sand.info())

# Print info on no_sand data
print(No_Sand.info())

# First rows of `No Sand` 
No_Sand.head()

# Last rows of `Sand`
Sand.tail()

# Take a sample of 5 rows of `No Sand`
No_Sand.sample(5)

# Describe `sand`
Sand.describe()

# Double check for null values in `No_sand`
pd.isnull(No_Sand)

# Add `type` column to `No Sand` with value 1
No_Sand['type'] = 1

# Add `type` column to `Sand` with value 0
Sand['type'] = 0

# Append `Sand` to `No_Sand`
Sanding = No_Sand.append(Sand, ignore_index=True)

import matplotlib.pyplot as plt

plt.hist(Sanding.Depth, 15, facecolor='blue')

plt.ylim([0, 6])
plt.xlabel("Depth (ft)")
plt.ylabel("Frequency")
plt.title("Distribution of Depth in ft")
plt.grid(True)
plt.show()

plt.hist(Sanding.Overburden, 10, facecolor='blue')
plt.ylim([0, 10])
plt.xlabel("Overburden (psi/ft)")
plt.ylabel("Frequency")
plt.title("Distribution of Overburden (psi/ft)")
plt.grid(True)
plt.show()

plt.hist(Sanding.Pore_Pressure, 15, facecolor='blue')
plt.ylim([0, 10])
plt.xlabel("Pore Pressure (psi/ft)")
plt.ylabel("Frequency")
plt.title("Distribution of Pore Pressure (psi/ft)")
plt.grid(True)
plt.show()

plt.hist(Sanding.Min_Horizontal_Stress, 2, facecolor='blue')
plt.ylim([0, 15])
plt.xlabel("Min Horizontal  Stress(psi/ft)")
plt.ylabel("Frequency")
plt.title("Distribution of Min Horizontal  Stress(psi/ft)")
plt.grid(True)
plt.show()

plt.hist(Sanding.Max_Horizontal_Stress, 2, facecolor='blue')
plt.ylim([0, 15])
plt.xlabel("Max Horizontal  Stress(psi/ft)")
plt.ylabel("Frequency")
plt.title("Distribution of Max Horizontal  Stress(psi/ft)")
plt.grid(True)
plt.show()

plt.hist(Sanding.Well_inclination, 3, facecolor='blue')
plt.ylim([0, 15])
plt.xlabel("Well Inclination (deg)")
plt.ylabel("Frequency")
plt.title("Distribution of Well Inclination (deg)")
plt.grid(True)
plt.show()

plt.hist(Sanding.well_azimuth, 2, facecolor='blue')
plt.ylim([0, 20])
plt.xlabel("Well Azimuth(deg)")
plt.ylabel("Frequency")
plt.title("Distribution of Well Azimuth(deg)")
plt.grid(True)
plt.show()

plt.hist(Sanding.poissons_ratio, 1, facecolor='blue')
plt.ylim([0, 30])
plt.xlabel("Poisson's Ratio")
plt.ylabel("Frequency")
plt.title("Distribution of Poisson's Ratio")
plt.grid(True)
plt.show()

plt.hist(Sanding.youngs_modulus, 5, facecolor='blue')
plt.ylim([0, 10])
plt.xlabel("Youngs Modulus(Mpsi)")
plt.ylabel("Frequency")
plt.title("Distribution of Young's Modulus(Mpsi)")
plt.grid(True)
plt.show()

plt.hist(Sanding.friction_angle, 8, facecolor='blue')
plt.ylim([0, 10])
plt.xlabel("Friction Angle(deg)")
plt.ylabel("Frequency")
plt.title("Distribution of Friction Angle(deg)")
plt.grid(True)
plt.show()

plt.hist(Sanding.shale_content, 15, facecolor='blue')
plt.ylim([0, 10])
plt.xlabel("Shale Content(%)")
plt.ylabel("Frequency")
plt.title("Distribution of Shale Content(%)")
plt.grid(True)
plt.show()

import seaborn as sns
corr = Sanding.corr()
# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))
# Set up the matplotlib figure
plt.subplots(figsize=(10, 6))
# Generate a custom diverging colormap
cmap = sns.diverging_palette(500, 100, as_cmap=True)
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

x = Sanding.iloc[:,0:11].values
y = Sanding.iloc[:,12].values
# Import `train_test_split` from `sklearn.model_selection`
from sklearn.model_selection import train_test_split

# Split the data up in train and test sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.75, random_state=42)

#PART 2: Building the ANN 

#Import Keras Libraries and Packages 
import keras
from keras.models import Sequential 
from keras.layers import Dense 

#Initializing the ANN
classifier = Sequential()

#Adding the input layer and the first hidden layer 
classifier.add(Dense(7, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))

#Adding the second hidden layer 
classifier.add(Dense(7, kernel_initializer = 'uniform', activation = 'relu'))

#Adding the output layer
classifier.add(Dense(1, kernel_initializer = 'uniform', activation = 'sigmoid'))

#Compiliing the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Fitting the ANN to the training set
history = classifier.fit(X_train, y_train,batch_size = 10, epochs = 200, validation_split=0.1, shuffle=False)

#PART 3: Making Predictions and Evaluating the Model

#Predicting the test set results 
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

score = classifier.evaluate(X_test, y_test,verbose=1)

print(score)

# Import the modules from `sklearn.metrics`
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print ("The confusion matrix is:\n", cm)
# Precision 
ps = precision_score(y_test, y_pred)
print ("The precision score is:\n", ps)
# Recall
rs = recall_score(y_test, y_pred)
print ("The recall score is:\n", rs)
# F1 score
f1 = f1_score(y_test,y_pred)
print ("The F1 Score is:\n", f1)
# Cohen's kappa
ck = cohen_kappa_score(y_test, y_pred)
print ("The Cohen Kappa score is:\n", ck)

"A Summarize History for Accuracy"
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


description = Sanding.describe()
print(description)






