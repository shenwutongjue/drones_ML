#!/usr/bin/env python
# coding: utf-8

# In[7]:


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


# In[8]:


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


# In[9]:


from sklearn.preprocessing import  StandardScaler
import pickle

standard_scaler = StandardScaler()
X_standardized = standard_scaler.fit_transform(X)

# Save the scalers
standard_file_path = "standard_scaler.pkl"


with open(standard_file_path, 'wb') as f:
    pickle.dump(standard_scaler, f)

print("Scaler are pickled and saved to", standard_file_path)

with open(standard_file_path, 'rb') as f:
    loaded_standard_scaler = pickle.load(f)

X_test_standardized = loaded_standard_scaler.transform(X_test)


# In[10]:


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


# In[11]:


#save the model
file_path = "best_SVM_1v1_model.pkl"

with open(file_path, 'wb') as f:
    pickle.dump(SVM, f)

print("SVM is pickled and saved to", file_path)


# In[ ]:




