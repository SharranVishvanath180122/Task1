#!/usr/bin/env python
# coding: utf-8

# In[2]:


#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


#load dataset
datapath="https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv"
data=pd.read_csv(datapath)


# In[4]:


#show the dataset
data


# In[5]:


#describe the dataset
data.describe()


# In[6]:


x=data.iloc[:,:-1].values
y=data.iloc[:,-1].values


# In[10]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[11]:


from sklearn.linear_model import LinearRegression
L=LinearRegression()
L.fit(x_train,y_train)


# In[13]:


Y_pred=L.predict(x_test)
L.predict([[9.25]])


# In[14]:


plt.scatter(x_train,y_train,color='green')
plt.plot(x_train,L.predict(x_train),color='Red')
plt.show()


# In[15]:


plt.scatter(x_test,y_test,color='blue')
plt.plot(x_train,L.predict(x_train),color='Red')
plt.show()

