#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import seaborn as sns
import matplotlib.pyplot as plt


# In[29]:


hand_df = pd.read_csv("data/sensorFile.csv", na_values=['?'])

#hand_df = hand_df._get_numeric_data()

hand_df = hand_df.fillna(0)
hand_df = hand_df.drop([0])
hand_df = hand_df.drop(['GyroX','GyroY','GyroZ'], axis = 1)

y = hand_df['Gesture']
X = hand_df.drop(['Gesture'], axis = 1)

print(X)


# In[31]:


# data preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#Standarization
min_max_scaler = MinMaxScaler()
X_min_max_scaled = min_max_scaler.fit_transform(X)
standard_scaler = StandardScaler()
X_standardized = standard_scaler.fit_transform(X)

#Check for feature dependency
correlation_matrix = hand_df.corr()

import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.show()


# In[60]:


# Check for datapoint similarity

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)

# Create a DataFrame with standardized features
data_standardized = pd.DataFrame(X_standardized, columns=['Thumb', 'Index', 'Middle', 'Ring', 'Pinky'])
data_standardized['class'] = y

sns.pairplot(data_standardized, hue='class', palette='viridis')
plt.show()


# In[6]:


from sklearn.preprocessing import MinMaxScaler, StandardScaler

train_size = 0.75

# Spliting data

np.random.seed(0)

X_train, X_rem, y_train, y_rem = train_test_split(X_standardized,y, train_size=train_size,random_state = 0)
test_size = 0.5
X_valid, X_test, y_valid, y_test = train_test_split(X_rem,y_rem, test_size=0.5,random_state = 0)

highest_valid = 0
accuracy_array = []

for k in range (1, 33+1, 2): 
    KNN_Model = KNeighborsClassifier(n_neighbors=k)
    KNN_Model.fit(X_train, y_train)

    y_predict_train = KNN_Model.predict(X_train)
    y_predict_valid = KNN_Model.predict(X_valid)
    y_predict_test = KNN_Model.predict(X_test)


    acc_train = accuracy_score(y_train, y_predict_train)
    acc_valid = accuracy_score(y_valid, y_predict_valid)
    acc_test = accuracy_score(y_test, y_predict_test)
    
    print('\n',k, '\t', acc_train, '\t', acc_valid, '\t', acc_test)

    if acc_valid > highest_valid:
        highest_valid = acc_valid
    
    accuracy_array.append([k, acc_valid])


for j in accuracy_array:
    if j[1] == highest_valid:
        print("\nbest k: ", j[0], "\thigest validation accuracy: ", j[1])
        
    


# In[7]:


#Decision Tree

dts = DecisionTreeClassifier(random_state = 0)

dts.fit(X_standardized, y)

max_depth = dts.get_depth()
max_leaves = dts.get_n_leaves()

print (max_depth, '\t', max_leaves)
plt.figure()
plot_tree(dts, filled=True)
plt.title("Decision tree trained on all features")
plt.show()


# In[8]:


np.random.seed(0)
# keep some data out for testing train/test 75/25
X_train, X_test, y_train, y_test = train_test_split(X_standardized,y, train_size=0.75,random_state = 0)

best_cvs=0
best_d=0

for d in range (1, max_depth+1): 
    dts = DecisionTreeClassifier(max_depth = d, random_state = 0)
    dts.fit(X_train, y_train)
    print("\nd: ", d)
    cvs = cross_val_score(dts, X_train, y_train, cv=10)
    print("Mean Cross-Validation Accuracy:", np.mean(cvs))
    test_score = dts.score(X_test, y_test)
    print("Test Set Accuracy:", test_score)
    if best_cvs<np.mean(cvs):
        best_cvs=np.mean(cvs)
        best_d=d
 
print('\n\nnbest d', '\t', 'higest cross validation score\n')
print(best_d, '\t', best_cvs, '\n')


# In[17]:


X_train, X_test, y_train, y_test = train_test_split(X_standardized,y, train_size=0.75,random_state = 0)
svm = SVC(kernel='linear') 
svm.fit(X_train, y_train)

cvs = cross_val_score(clf, X_standardized, y, cv = 10)

print("default SVC cvs:",cvs)


# In[72]:


#svm hyper-paramerter tunning

from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.001, 0.01, 0.1, 0.5, 1, 5, 10, 100],      # Regularization parameter
    'kernel': ['linear', 'rbf'], # Kernel type
    'gamma': ['scale', 'auto'],  # Kernel coefficient
}

grid_search = GridSearchCV(svm, param_grid, cv=10, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Get the results as a DataFrame
results_df = pd.DataFrame(grid_search.cv_results_)

# Print all parameters and corresponding mean test scores
for index, row in results_df.iterrows():
    params = row['params']
    mean_test_score = row['mean_test_score']
    
    print("Parameters:", params)
    print("Mean Test Score:", mean_test_score)
    
# Get the best parameters from the grid search
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)


# Pred with best params
best_svm_model = grid_search.best_estimator_
y_pred = best_svm_model.predict(X_test)

# Evaluate the performance on the test set
test_accuracy = accuracy_score(y_test, y_pred)
print("Test Set Accuracy:", test_accuracy)


# In[71]:


#One versus Rest
from sklearn.multiclass import OneVsRestClassifier

X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.75,random_state = 0)

# Define the base classifier
base_classifier = SVC()

# Define the OneVsRestClassifier
clf = OneVsRestClassifier(base_classifier)

# Define the parameter grid to search
param_grid = {
    'estimator__C': [0.001, 0.01, 0.1, 0.5, 1, 5, 10, 100],  # Adjust the range based on your needs
    'estimator__kernel': ['linear', 'rbf'],  # You can try other kernels as well
}

# Create the grid search object
grid_search = GridSearchCV(clf, param_grid, cv=10, scoring='accuracy', n_jobs=-1)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Get the results as a DataFrame
results_df = pd.DataFrame(grid_search.cv_results_)

# Print all parameters and corresponding mean test scores
for index, row in results_df.iterrows():
    params = row['params']
    mean_test_score = row['mean_test_score']
    
    print("Parameters:", params)
    print("Mean Test Score:", mean_test_score)

# Get the best parameters from the grid search
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Get the best model from the grid search
best_model = grid_search.best_estimator_

# Evaluate the model on the test set
test_score = best_model.score(X_test, y_test)
print("Test Set Accuracy:", test_score)


# In[ ]:




