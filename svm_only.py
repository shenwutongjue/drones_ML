#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
#import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsOneClassifier

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import pickle


# In[3]:


hand_df = pd.read_csv("data/sensorFile (2).csv", na_values=['?'])
hand_df_test = pd.read_csv("data/Newglove_person1.csv", na_values=['?'])


hand_df = hand_df.fillna(0)
hand_df = hand_df.drop([0])
hand_df = hand_df.drop(['GyroX','GyroY','GyroZ'], axis = 1)

hand_df_test = hand_df_test.fillna(0)
hand_df_test = hand_df_test.drop([0])
hand_df_test = hand_df_test.drop(['GyroX','GyroY','GyroZ'], axis = 1)

y = hand_df['Gesture']
X = hand_df.drop(['Gesture'], axis = 1)

y_test = hand_df_test['Gesture']
X_test = hand_df_test.drop(['Gesture'], axis = 1)


# In[4]:


# data preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#Standarization
min_max_scaler = MinMaxScaler()
X_min_max_scaled = min_max_scaler.fit_transform(X)
standard_scaler = StandardScaler()
X_standardized = standard_scaler.fit_transform(X)

#save the scaler
file_path = "standard_scaler.pkl"

with open(file_path, 'wb') as f:
    pickle.dump(standard_scaler, f)

print("StandardScaler object is pickled and saved to", file_path)

scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)
X_test_standardized = scaler.fit_transform(X_test)


# In[14]:


#SVM with best params

X_test = X_test_standardized

best_params = {'C': 10, 'kernel': 'rbf', 'gamma': 'scale'}
SVM = OneVsOneClassifier(SVC(**best_params))

SVM.fit(X_standardized,y)


y_pred_train = SVM.predict(X_standardized)
y_pred_test = SVM.predict(X_test)
train_accuracy = accuracy_score(y, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)
print(f"\nTrain Set Accuracy with Best Hyperparameters: {train_accuracy}")
print(f"\nTest Set Accuracy with Best Hyperparameters: {test_accuracy}")


# In[15]:


#save the model
file_path = "best_SVM_1v1_model.pkl"

with open(file_path, 'wb') as f:
    pickle.dump(SVM, f)

print("SVM is pickled and saved to", file_path)


# In[ ]:




