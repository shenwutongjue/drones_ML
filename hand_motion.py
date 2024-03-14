#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

import pickle


# In[2]:


get_ipython().system('python --version')


# In[15]:


#hand_df = pd.read_csv("data/sensorFile.csv", na_values=['?'])
hand_df = pd.read_csv("data/sensorFile (2).csv", na_values=['?'])
#hand_df_test = pd.read_csv("data/sensorFile_Michael.csv", na_values=['?'])
hand_df_test = pd.read_csv("data/Newglove_person1.csv", na_values=['?'])

#hand_df = hand_df._get_numeric_data()

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


print(X)


# In[16]:


# data preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#Standarization
standard_scaler = StandardScaler()
X_standardized = standard_scaler.fit_transform(X)

# Save the scalers
standard_file_path = "standard_scaler.pkl"


with open(standard_file_path, 'wb') as f:
    pickle.dump(standard_scaler, f)

print("Scaler are pickled and saved to", standard_file_path)

import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.show()


# In[17]:


#Testing Standardization for Smaller Inputs

hand_df_short = pd.read_csv("data/sensorFile_Michael_test.csv", na_values=['?'])
hand_df_short = hand_df_short.fillna(0)
hand_df_short = hand_df_short.drop([0])


hand_df_short = hand_df_short.drop(['GyroX','GyroY','GyroZ'], axis = 1)

y_test1 = hand_df_short['Gesture']
X_test1 = hand_df_short.drop(['Gesture'], axis = 1)

print(X_test1)

X_standardized1 = standard_scaler.fit_transform(X_test1)

print("\n",X_standardized1)


# In[18]:


#save the scaler
import pickle

file_path = "standard_scaler.pkl"

with open(file_path, 'wb') as f:
    pickle.dump(standard_scaler, f)

print("StandardScaler object is pickled and saved to", file_path)


# In[19]:


# Check for datapoint similarity

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)
X_test_standardized = scaler.fit_transform(X_test)

# Create a DataFrame with standardized features
data_standardized = pd.DataFrame(X_standardized, columns=['Thumb', 'Index', 'Middle', 'Ring', 'Pinky'])
data_standardized['class'] = y

sns.pairplot(data_standardized, hue='class', palette='viridis')
plt.show()


# In[10]:


from sklearn.preprocessing import MinMaxScaler, StandardScaler

best_k = None
best_accuracy = 0

X_train, X_valid, y_train, y_valid = train_test_split(X_standardized,y, train_size=0.75, random_state = 0)


X_test = X_test_standardized


print('\n k\tacc_train\tacc_valid\tacc_test\tMean CVS')

for k in range(1, 33+1, 2):
    KNN_Model = KNeighborsClassifier(n_neighbors=k)
    KNN_Model.fit(X_train, y_train)

    y_predict_valid = KNN_Model.predict(X_valid)
    y_predict_train = KNN_Model.predict(X_train)
    y_predict_test = KNN_Model.predict(X_test)
    acc_valid = accuracy_score(y_valid, y_predict_valid)
    acc_train = accuracy_score(y_train, y_predict_train)
    acc_test = accuracy_score(y_test, y_predict_test)

    if acc_valid > best_accuracy:
        best_accuracy = acc_valid
        best_k = k

    cvs = cross_val_score(KNN_Model, X_train, y_train, cv=10)
    
    print('\n', k, '\t', acc_train, '\t', acc_valid, '\t', acc_test, '\t', np.mean(cvs))

print('\nBest k:', best_k)

# Now use the best_k to train the model and make predictions
best_model = KNeighborsClassifier(n_neighbors=best_k)
best_model.fit(X_train, y_train)

y_predict_train = best_model.predict(X_train)
y_predict_valid = best_model.predict(X_valid)
y_predict_test = best_model.predict(X_test)

acc_train = accuracy_score(y_train, y_predict_train)
acc_valid = accuracy_score(y_valid, y_predict_valid)
acc_test = accuracy_score(y_test, y_predict_test)

print('\nAccuracy on Train set:', acc_train)
print('Accuracy on Validation set:', acc_valid)
print('Accuracy on Test set:', acc_test)


# In[10]:


# Pickle the best model, you can use the same method to pickle other models
file_path = "best_knn_model.pkl"
with open(file_path, 'wb') as f:
    pickle.dump(best_model, f)

print("Best model pickled and saved to", file_path)


# In[11]:


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


# In[12]:


# tunning tree depth

X_train = X_standardized
y_train = y

X_test = X_test_standardized

best_cvs=0
best_d=0

print("\nd\tMean CVS\tTest Set Accuracy\n")

for d in range (1, max_depth+1): 
    dts = DecisionTreeClassifier(max_depth = d, random_state = 0)
    dts.fit(X_train, y_train)
    cvs = cross_val_score(dts, X_train, y_train, cv=10)
    test_score = dts.score(X_test, y_test)
    print(d,'\t',np.mean(cvs),'\t',test_score)
    if best_cvs<np.mean(cvs):
        best_cvs=np.mean(cvs)
        best_d=d
 
print('\n\nbest d', '\t', 'higest cross validation score\n')
print(best_d, '\t', best_cvs, '\n')

# train dts on best d
dts = DecisionTreeClassifier(max_depth = best_d, random_state = 0)
dts.fit(X_standardized, y)

plt.figure()
plot_tree(dts, filled=True)
plt.title(f"Decision tree trained on d = {best_d}")
plt.show()


# In[13]:


X_train, X_valid, y_train, y_valid = train_test_split(X_standardized,y, train_size=0.75, random_state = 0)


X_test = X_test_standardized

svm = SVC(kernel='linear') 
svm.fit(X_train, y_train)

cvs = cross_val_score(svm, X_standardized, y, cv = 10)

print("default SVC cvs:",cvs)


# In[14]:


# Define hyperparameters to try
C_values = [0.001, 0.01, 0.1, 0.5, 1, 5, 10, 100]
kernel_values = ['linear', 'rbf']
gamma_values = ['scale', 'auto']

best_accuracy = 0
best_params = None

# Iterate over hyperparameter combinations
for C in C_values:
    for kernel in kernel_values:
        for gamma in gamma_values:
            svm = SVC(C=C, kernel=kernel, gamma=gamma)
            svm.fit(X_train, y_train)


            y_pred_valid = svm.predict(X_valid)
            valid_accuracy = accuracy_score(y_valid, y_pred_valid)

            print(f"C={C}, Kernel={kernel}, Gamma={gamma}, Validation Accuracy={valid_accuracy}")

            if valid_accuracy > best_accuracy:
                best_accuracy = valid_accuracy
                best_params = {'C': C, 'kernel': kernel, 'gamma': gamma}

# Train the final model with the best hyperparameters on the combined training and validation sets
final_svm = SVC(**best_params)
final_svm.fit(np.concatenate((X_train, X_valid)), np.concatenate((y_train, y_valid)))

# Evaluate on the test set
y_pred_test = final_svm.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred_test)
print(f"\nTest Set Accuracy with Best Hyperparameters: {test_accuracy}")
print("Best Hyperparameters:", best_params)


# In[15]:


#1 vs. Rest
from sklearn.multiclass import OneVsRestClassifier

X_train = X_standardized
y_train = y

X_test = X_test_standardized

# Define hyperparameters to try
C_values = [0.001, 0.01, 0.1, 0.5, 1, 5, 10, 100]
kernel_values = ['linear', 'rbf']
gamma_values = ['scale', 'auto']

best_accuracy = 0
best_params = None

# Iterate over hyperparameter combinations
for C in C_values:
    for kernel in kernel_values:
        for gamma in gamma_values:
            clf = OneVsRestClassifier(SVC(C=C, kernel=kernel, gamma=gamma))
            clf.fit(X_train, y_train)

            y_pred_valid = clf.predict(X_valid)
            valid_accuracy = accuracy_score(y_valid, y_pred_valid)

            print(f"C={C}, Kernel={kernel}, Gamma={gamma}, Validation Accuracy={valid_accuracy}")

            if valid_accuracy > best_accuracy:
                best_accuracy = valid_accuracy
                best_params = {'C': C, 'kernel': kernel, 'gamma': gamma}

# Train the final model with the best hyperparameters on the combined training and validation sets
final_clf = OneVsRestClassifier(SVC(**best_params))
final_clf.fit(np.concatenate((X_train, X_valid)), np.concatenate((y_train, y_valid)))

# Evaluate on the test set
y_pred_test = final_clf.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred_test)
print(f"\nTest Set Accuracy with Best Hyperparameters: {test_accuracy}")
print("Best Hyperparameters:", best_params)


# In[23]:


# 1 vs. 1
from sklearn.multiclass import OneVsOneClassifier

X_train, X_valid, y_train, y_valid = train_test_split(X_standardized,y, train_size=0.75, random_state = 0)

X_test = X_test_standardized

# Define hyperparameters to try
C_values = [0.001, 0.01, 0.1, 0.5, 1, 5, 10, 100]
kernel_values = ['linear', 'rbf']
gamma_values = ['scale', 'auto']

best_accuracy = 0
best_params = None

# Iterate over hyperparameter combinations
for C in C_values:
    for kernel in kernel_values:
        for gamma in gamma_values:
            clf1 = OneVsOneClassifier(SVC(C=C, kernel=kernel, gamma=gamma))
            clf1.fit(X_train, y_train)
            y_pred_valid = clf1.predict(X_valid)
            valid_accuracy = accuracy_score(y_valid, y_pred_valid)

            print(f"C={C}, Kernel={kernel}, Gamma={gamma}, Validation Accuracy={valid_accuracy}")

            if valid_accuracy > best_accuracy:
                best_accuracy = valid_accuracy
                best_params = {'C': C, 'kernel': kernel, 'gamma': gamma}

# Train the final model with the best hyperparameters on the combined training and validation sets
final_clf1 = OneVsOneClassifier(SVC(**best_params))
final_clf1.fit(np.concatenate((X_train, X_valid)), np.concatenate((y_train, y_valid)))

# Evaluate on the test set
y_pred_test = final_clf1.predict(X_test)
y_pred_train = final_clf1.predict(X_train)
y_pred_valid = final_clf1.predict(X_valid)
train_accuracy = accuracy_score(y_train, y_pred_train)
valid_accuracy = accuracy_score(y_valid, y_pred_valid)
test_accuracy = accuracy_score(y_test, y_pred_test)
print(f"\nTrain Set Accuracy with Best Hyperparameters: {train_accuracy}")
print(f"\nValid Set Accuracy with Best Hyperparameters: {valid_accuracy}")
print(f"\nTest Set Accuracy with Best Hyperparameters: {test_accuracy}")
print("Best Hyperparameters:", best_params)



# In[29]:


best_params = {'C': 10, 'kernel': 'rbf', 'gamma': 'scale'}
final_clf1 = OneVsOneClassifier(SVC(**best_params))

final_clf1.fit(X_standardized,y)

file_path = "best_SVM_1v1_model.pkl"
with open(file_path, 'rb') as f:
    loaded_model = pickle.load(f)

y_pred_train = loaded_model.predict(X_standardized)
y_pred_test = loaded_model.predict(X_test)
train_accuracy = accuracy_score(y, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)
print(f"\nTrain Set Accuracy with Best Hyperparameters: {train_accuracy}")
print(f"\nTest Set Accuracy with Best Hyperparameters: {test_accuracy}")


# In[25]:


#model persistance
from tempfile import mkdtemp
import joblib

savedir = mkdtemp()

import os
filename = os.path.join(savedir, 'OneVsRest.joblib')
joblib.dump(final_clf, filename)  


# In[8]:


from joblib import load

# Specify the path to the joblib file
file_path = 'OneVsRest.joblib'

# Load the joblib file
loaded_data = load(file_path)


# In[ ]:




y_pred_test = loaded_data.predict(X_test_input)


# In[4]:


loaded_svc = joblib.load(filename)
y_pred_test = loaded_svc.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred_test)
print(f"\nTest Set Accuracy with Best Hyperparameters: {test_accuracy}")


# In[ ]:


filename1 = os.path.join(savedir, 'dts.joblib')
joblib.dump(dts, filename1)  


# In[ ]:


loaded_svc = joblib.load(filename1)
y_pred_test = loaded_svc.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred_test)
print(f"\nTest Set Accuracy with Best Hyperparameters: {test_accuracy}")


# In[ ]:


filename2 = os.path.join(savedir, 'OneVsOne.joblib')
joblib.dump(final_clf1, filename2)  


# In[ ]:


loaded_svc = joblib.load(filename2)
y_pred_test = loaded_svc.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred_test)
print(f"\nTest Set Accuracy with Best Hyperparameters: {test_accuracy}")


# In[ ]:


filename3 = os.path.join(savedir, 'SVM.joblib')
joblib.dump(final_svm, filename3)  


# In[ ]:


loaded_svc = joblib.load(filename3)
y_pred_test = loaded_svc.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred_test)
print(f"\nTest Set Accuracy with Best Hyperparameters: {test_accuracy}")


# In[ ]:


filename4 = os.path.join(savedir, 'KNN.joblib')
joblib.dump(best_model, filename4)  


# In[53]:


loaded_svc = joblib.load(filename4)
y_pred_test = loaded_svc.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred_test)
print(f"\nTest Set Accuracy with Best Hyperparameters: {test_accuracy}")


# In[ ]:




