#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as data
import yfinance as yf


# In[ ]:


#We can obtain predicted values of any stock listed company with changing the ticker symbol.


# In[84]:


#setting the start date & end date
start= '2009-12-31'
end = '2019-12-31'

#Taking data from yf the data package from python library
df = yf.download('AAPL',start,end)

#displaying starting readings
df


# In[85]:


#displaying ending few readings
df.tail()


# In[86]:


#reset_index is used to initilize editing to the table
df = df.reset_index()
df.head()


# In[87]:


#dropping/deleting Date & Adj Close columns
df = df.drop(['Date' , 'Adj Close'],axis= 1)
df.head()


# In[118]:


#Ploting the above readings in a line graph
plt.figure(figsize=(15,8))
plt.plot(df.Close)


# In[89]:


#All readings & it's count
df


# In[90]:


# ma is for moving average 
# It will calculate the moving average for first 100 readings
#First 100 values will be null values
ma100 = df.Close.rolling(100).mean()
ma100


# In[119]:


# Setting the dimentions of the graph 
plt.figure(figsize =(15,8))

#Plotting Close & ma100 in the graph
plt.plot(df.Close)

#Plotting ma100 with red colour line
plt.plot(ma100, 'r')


# In[92]:


# It will calculate the moving average for first 200 readings
#First 200 values will be null values
ma200 = df.Close.rolling(200).mean()
ma200


# In[120]:


plt.figure(figsize =(15,8))

#Plotting Close, ma100 & ma200 in the graph
plt.plot(df.Close)

#Plotting ma100 with red colour line & ma200 with green line
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')


# In[94]:


#This shows us that there are 2516 rows & 5 columns in our data set
df.shape


# In[95]:


#Splitting data into training & testing

#Creating DataFrame for Training & Testing 
#Staring with 0 index we will take 70% of the value for Training
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

#Printing the shapes of Testing & Training
print(data_training.shape)
print(data_testing.shape)


# In[96]:


#Starting few readings of the data_training set
data_training.head()


# In[97]:


#Starting few readings of the data_testing set
data_testing.head()


# In[98]:


#from the sklearn library we import a specific library i.e. MinMaxScaler for scaling down the data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))


# In[99]:


#Converting data_training into an array
data_training_array = scaler.fit_transform(data_training)
data_training_array


# In[100]:


x_train = []
y_train = []

#x_train will start with the 0th index  
#y_train will start from i
for i in range(100, data_training_array.shape[0]):
    x_train.append(data_training_array[i-100: i])
    y_train.append(data_training_array[i, 0])
    
x_train, y_train = np.array(x_train), np.array(y_train)


# In[101]:


#ML model
#Keras is a minimalist Python library for deep learning
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential


# In[104]:


# Dropout works by randomly setting the outgoing edges of hidden units
#Dropout is a technique used to prevent a model from overfitting.
model = Sequential()
model.add(LSTM(units = 50, activation = 'relu', return_sequences = True,
              input_shape = (x_train.shape[1],1)))
model.add(Dropout(0.2))

model.add(LSTM(units = 60, activation = 'relu', return_sequences = True))
model.add(Dropout(0.3))

model.add(LSTM(units = 80, activation = 'relu', return_sequences = True))
model.add(Dropout(0.4))

model.add(LSTM(units = 120, activation = 'relu'))
model.add(Dropout(0.5))

#Connects all the above layers
model.add(Dense(units = 1))


# In[105]:


#Defines the summary of the data
model.summary()


# In[36]:


#Compiling the ML model with adam optimizer with loss as mean_squared_error
model.compile(optimizer ='adam', loss= 'mean_squared_error')

#epoch is an instant of time or a date selected as a point of reference
model.fit(x_train, y_train, epochs = 50)


# In[38]:


data_testing.head()


# In[106]:


#Fetching past 100 days from data_training set
past_100_days = data_training.tail(100)


# In[107]:


#appending past 100 days to data_testing
final_df = past_100_days.append(data_testing, ignore_index=True)


# In[41]:


final_df.head()


# In[108]:


#Scaling down final_df
input_data = scaler.fit_transform(final_df)
input_data


# In[109]:


#Shape of input_data
input_data.shape


# In[46]:


#Declaring the x_test & y_test & appending it to input_data
x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])


# In[47]:


#Converting to numpy array
x_test, y_test = np.array(x_test), np.array(y_test)
print(x_test)
print(y_test)


# In[49]:


#Making predictions
y_predicted = model.predict(x_test)


# In[50]:


#Shape of y_predicted
y_predicted.shape


# In[51]:


#Displaying the array of y_test
y_test


# In[52]:


y_predicted


# In[53]:


#Finding the factor by which we scaled down the data
scaler.scale_


# In[55]:


#Dividing y_test & y_predicted with the scale factor
scale_factor = 1/ 0.02123255
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor


# In[75]:


plt.figure(figsize=(15,8))
#Plotting y_predicted with red colour line 
plt.plot(y_predicted, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()


# In[76]:


plt.figure(figsize=(15,8))
#Plotting y_predicted with blue colour line 
plt.plot(df.Close, 'b' , label= 'Original Price')


# In[117]:


#This show the difference between the values of the original price & the predicted price
df['price_diff'] = df['Close'].diff()
df

