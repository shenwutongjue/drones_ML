#!/usr/bin/env python
# coding: utf-8

# In[1]:


#load model and scaler, change model name if using other pickeled models
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier

file_path = "best_knn_model.pkl"
with open(file_path, 'rb') as f:
    loaded_model = pickle.load(f)
    
file_path = "standard_scaler.pkl"
with open(file_path, 'rb') as f:
    scaler = pickle.load(f)


# In[3]:


#predict
from serial import*
from collections import deque 
#ser.close()
serialPort = 'COM3'
baudRate = 19200
ser = Serial(serialPort,baudRate,timeout=1)
inputBuffer = deque()

try:
    data = ser.readline()
    while True:
        data = ser.readline()
        #GyroXRaw = 0
        #GyroYRaw = 0
        #GyroZRaw = 0
        flexThumb = 0
        flexIndex = 0
        flexMiddle = 0
        flexRing = 0
        flexPinky = 0
        try:
            #GyroXRaw = int(data[2:8].decode('utf-8'))
            #GyroYRaw = int(data[10:16].decode('utf-8'))
            #GyroZRaw = int(data[18:23].decode('utf-8'))
            flexThumb = int(data[27:33].decode('utf-8'))
            flexIndex = int(data[37:43].decode('utf-8'))
            flexMiddle = int(data[47:53].decode('utf-8'))
            flexRing = int(data[57:63].decode('utf-8'))
            flexPinky = int(data[67:73].decode('utf-8'))
            
            flexList = [flexThumb, flexIndex, flexMiddle, flexRing, flexPinky]
            inputBuffer.append(flexList)
            #print( "F1: ", flexThumb, "F2: ", flexIndex, "F3: ", flexMiddle, "F4: ", flexRing, "F5: ", flexPinky)
            #print(flexList)
            if len(inputBuffer) >= 20:
                inputBuffer.pop()
                print("predicting...")
                X_Input_standardized = scaler.fit_transform(inputBuffer)
                #print("standardized")
                #print(X_Input_standardized)
                y_pred_test = loaded_model.predict(X_Input_standardized)
                print("finished predicting")
                print(y_pred_test)
                #inputBuffer = deque()
                
                
        except Exception:
            pass
        
        #print( "F1: ", flexThumb, "F2: ", flexIndex, "F3: ", flexMiddle, "F4: ", flexRing, "F5: ", flexPinky)
        
except KeyboardInterrupt:
    print("keyboard interrupt")
finally:
    ser.close()


# In[ ]:


#rest of the code are for model testing with hard-coded data


# In[14]:


from collections import deque

num_rows = 20
num_cols = 5

data = [
    [32435, 11403, 12060, 11868, 15710],
    [32369, 11416, 12073, 11792, 15805],
    [32402, 11428, 12009, 11766, 15837],
    [32303, 11465, 12060, 11855, 15710],
    [32369, 11465, 11970, 11766, 15663],
    [32271, 11453, 11893, 11817, 15710],
    [32238, 11453, 11804, 11741, 15663],
    [32271, 11403, 11766, 11653, 15710],
    [32140, 11391, 11653, 11703, 15773],
    [32369, 11354, 11728, 11754, 16093],
    [19708, 21458, 23777, 20275, 10771],
    [19689, 21375, 23777, 20315, 10819],
    [19477, 21375, 23777, 20355, 10843],
    [19024, 21375, 23707, 20315, 10867],
    [18654, 21458, 23730, 20315, 10783],
    [18399, 21375, 23800, 20395, 10771],
    [17951, 21375, 23777, 20256, 10831],
    [17050, 21270, 23940, 20395, 10855],
    [16983, 21375, 23940, 20355, 10831],
    [17016, 21312, 23754, 20275, 10819]
]

X_input = deque(maxlen=num_rows)

for row in data:
    X_input.append(row)

for row in X_input:
    print(row)


# In[16]:


X_Input_standardized = scaler.fit_transform(X_input)
#X_Input_standardized1 = scaler.fit_transform(X_input1)


# In[5]:


X_Input_standardized


# In[6]:


y_pred_test = loaded_model.predict(X_Input_standardized)
#y_pred_test1 = model.predict(X_Input_standardized1)


# In[7]:


y_pred_test


# In[1]:


print(inputBuffer)


# In[15]:


print(X_input)


# In[23]:


data = [[ 0.60793035, 0.57354901, -0.16302764,  0.07988306, -1.31953627],
 [ 0.74135401,0.57354901,  0.59944008, -2.22489102,  2.16060047],
 [ 0.60793035,-1.21969916,  1.31425357, -0.70711297,  2.16060047],
 [ 0.53931247, 0.57354901,  1.31425357,  0.07988306,  0.4560437 ],
 [ 0.74135401, 0.57354901,  0.59944008, -0.70711297, -0.46725789],
 [ 0.74135401,-1.21969916, -0.16302764,  0.07988306,  2.16060047],
 [ 0.60793035, 0.57354901, -0.16302764,  0.07988306, -0.46725789],
 [ 0.74135401, 0.57354901, -0.16302764,  0.86687908, -0.46725789],
 [ 0.74135401,  0.57354901,  2.07672128, -2.22489102, -0.46725789],
 [ 0.80997189, -1.21969916, -0.16302764, -0.70711297, -0.46725789],
 [ 0.74135401,  0.57354901, -0.16302764, 0.86687908, -0.46725789],
 [ 0.60793035, -1.21969916, -0.16302764,  0.86687908, -0.46725789],
 [ 0.47450669, -1.21969916, -0.16302764,  0.86687908, -0.46725789],
 [-1.38580031, -1.21969916, -0.16302764, -0.70711297, -0.46725789],
 [-1.90805862,  0.57354901, -0.16302764,  0.86687908, -0.46725789],
 [-1.64502341, -1.21969916, -0.16302764,  0.07988306, -0.46725789],
 [-1.38580031,  2.22885501, -0.16302764,  1.59766111, -0.46725789],
 [-1.32099453, 0.57354901, -0.87784112,  0.07988306,  0.4560437 ],
 [-1.05795932,  0.57354901, -3.06993581,  0.86687908, -0.46725789]]

X_input = deque(maxlen=num_rows)

for row in data:
    X_input.append(row)

for row in X_input:
    print(row)


# In[28]:


y_pred_test = loaded_model.predict(X_input)


# In[29]:


y_pred_test


# In[ ]:




