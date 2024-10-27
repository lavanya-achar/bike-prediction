#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[3]:


import numpy as np


# In[5]:


df = pd.read_csv('https://raw.githubusercontent.com/Lorddhaval/Dataset/patch-1/Bike%20Prices.csv')
     


# In[7]:


df.head()


# In[9]:


df.describe()


# In[11]:


df.info()


# In[13]:


df[['Brand']].value_counts()


# In[15]:


df[['Model']].value_counts()


# In[17]:


df.columns
     


# In[21]:


df.replace({'Seller_Type':{'Individual':0, 'Dealer':1}},inplace=True)
     


# In[23]:


df.replace({'Owner':{'1st owner':0, '2nd owner' :1, '3rd owner':2, '4th owner':3}},inplace=True)
     


# In[25]:


y = df['Selling_Price']
     

y.shape
     


y


# In[29]:


X = df[['Year', 'Seller_Type', 'Owner' ,'KM_Driven','Ex_Showroom_Price']]
     

X.shape
     
X



# In[31]:


from sklearn.model_selection import train_test_split
     

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=222529)
     

X_train.shape , X_test.shape, y_train.shape, y_test.shape
   


# In[33]:


from sklearn.linear_model import LinearRegression
     

lr = LinearRegression()
     

lr.fit(X_train, y_train)


# In[35]:


y_pred = lr.predict(X_test)
     

y_pred.shape
     
(188,)

y_pred
     


# In[39]:


from sklearn.metrics import mean_squared_error , mean_absolute_error , r2_score
     

mean_squared_error(y_test,y_pred)
     





# In[41]:


mean_absolute_error(y_test,y_pred)
     




# In[43]:


r2_score(y_test,y_pred)


# In[45]:


import matplotlib.pyplot as plt
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Price vs Predicted Price") 
plt.show()
     


# In[47]:


df_new = df.sample(1)
     

df_new


# In[49]:


X_new = df_new.drop(['Brand', 'Model', 'Selling_Price'], axis = 1)
     



# In[51]:


y_pred_new = lr.predict(X_new)
     


     


# In[53]:


y_pred_new


# In[3]:





# In[ ]:




