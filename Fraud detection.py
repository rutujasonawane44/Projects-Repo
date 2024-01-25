#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 
# import os
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly import tools
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)


# In[2]:


from sklearn.model_selection import train_test_split


# In[3]:


df=pd.read_csv(r'Downloads/Fraud.csv')


# In[4]:


df


# In[5]:


df.head(10)


# In[6]:


df.info()


# In[7]:


df.describe()


# In[8]:


df.isnull().sum() #there is no any null or missing value


# In[9]:


# data encoding
df.info()


# In[10]:


df


# In[11]:


#name origin and name Dest ---(drop)() see later::


# In[12]:



df1=df.drop(['nameOrig','nameDest'],axis=1)


# In[13]:


df1


# In[14]:


#encoding:
df1['type'].value_counts()


# In[15]:


df.nunique()


# In[16]:


# data exploration
type=df1['type'].value_counts()


# In[17]:



transactions=type.index


# In[18]:


quantity=type.values


# In[19]:



import plotly.express as px
px.pie(df1,values=quantity,names=transactions,hole=0.4,title="Distribution of Transaction Type")


# In[20]:


df['type'].unique()


# In[21]:


df1['type']=df1['type'].map({'PAYMENT':1,'TRANSFER':2,'CASH_OUT':3,'DEBIT':4,'CASH_IN':5})


# In[22]:


df1['type']


# In[23]:


df1


# In[24]:


df1['isFraud'].unique()


# In[25]:


df1['isFraud'].value_counts()


# In[26]:


sns.countplot(df1['isFraud'])


# In[27]:


get_ipython().system('pip install imblearn')


# In[28]:


from imblearn.over_sampling import RandomOverSampler


# In[29]:


obj = RandomOverSampler(sampling_strategy=0.70)


# In[30]:


x=df1.iloc[:,0:7]


# In[31]:


x


# In[32]:


df1


# In[45]:


y=df1['isFraud']


# In[ ]:





# In[34]:


X_new,y_new = obj.fit_resample(x,y)


# In[35]:


y.value_counts()


# In[36]:


y_new.value_counts()


# In[37]:


sns.countplot(y_new)


# In[ ]:





# In[38]:


df1['isFraud'].value_counts()


# In[39]:


# finding outliers:


# In[40]:


# Box Plot
import seaborn as sns
sns.boxplot(df1['amount'])


# In[49]:


sns.boxplot(data=df1,y='amount',x='type')
plt.show()


# In[42]:


sns.boxplot(df1['oldbalanceOrg'])


# In[50]:


sns.boxplot(data=df1,y='oldbalanceOrg',x='type')
plt.show()


# In[51]:


sns.boxplot(df1['newbalanceOrig'])


# In[52]:


sns.boxplot(data=df1, y ='newbalanceOrig' , x ='type')


# In[53]:


sns.boxplot(data=df1, y ='isFraud' , x ='type')


# In[ ]:


## removing outlier by IQR


# In[54]:


percentile25 = df1['oldbalanceOrg'].quantile(0.25)
percentile75 = df1['oldbalanceOrg'].quantile(0.75)


# In[55]:


IQR = percentile75 - percentile25


# In[56]:


uper_limit = percentile75 + 1.5 * IQR


# In[57]:


lower_limit = percentile25 - 1.5 * IQR


# In[58]:


df2=df1[df1['oldbalanceOrg'] < uper_limit]


# In[59]:


df2


# In[60]:


df1[df1['oldbalanceOrg'] <  lower_limit]


# In[61]:


percentile25 = df1['newbalanceOrig'].quantile(0.25)
percentile75 = df1['newbalanceOrig'].quantile(0.75)


# In[62]:


IQR = percentile75 - percentile25


# In[63]:


uper_limit = percentile75 + 1.5 * IQR


# In[64]:


lower_limit = percentile25 - 1.5 * IQR


# In[65]:


df2=df1[df1['newbalanceOrig'] < uper_limit]


# In[66]:


df2


# In[ ]:




