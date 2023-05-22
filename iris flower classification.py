#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from warnings import filterwarnings


# In[2]:


iris=pd.read_csv("iris.csv")
print(iris)


# In[3]:


print(iris.shape)


# In[4]:


print(iris.describe())


# In[5]:


iris.head()


# In[ ]:


iris.tail(100)


# In[6]:


n = len(iris[iris['Species'] == 'versicolor'])
print("No of Versicolor in Dataset:",n)


# In[7]:


n1 = len(iris[iris['Species'] == 'virginica'])
print("No of Virginica in Dataset:",n1)


# In[8]:


n2 = len(iris[iris['Species'] == 'setosa'])
print("No of Setosa in Dataset:",n2)


# In[9]:


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.axis('equal')
l = ['Versicolor', 'Setosa', 'Virginica']
s = [50,50,50]
ax.pie(s, labels = l,autopct='%1.2f%%')
plt.show()


# In[10]:


iris.hist()
plt.show()


# In[11]:


sns.pairplot(iris,hue='Species');


# In[12]:


X = iris['Sepal.Length'].values.reshape(-1,1)
print(X)


# In[14]:


Y = iris['Sepal.Width'].values.reshape(-1,1)
print(Y)


# In[15]:


plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.scatter(X,Y,color='b')
plt.show()


# In[ ]:




