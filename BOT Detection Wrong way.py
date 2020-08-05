#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# In[2]:


data = pd.read_csv('ibm_data.csv')


# In[3]:


print('########## Dataset Infomarion ##########')
print ("Rows     : " ,data.shape[0])
print ("Columns  : " ,data.shape[1])
print ("\n########## Features ##########\n" ,data.columns.tolist())
print ("\n Total Missing values : ",data.isnull().sum().values.sum())
print("\n########## Details for missing values : ##########\n", data.isnull().sum())


# In[4]:


data.columns


# In[5]:


data['ip_addr']= data['ip_addr'].astype(str)
data['VISIT']= data['VISIT'].astype(int)
data['ENGD_VISIT']= data['ENGD_VISIT'].astype(int)
data['VIEWS']= data['VIEWS'].astype(int)
data['wk']= data['wk'].astype(int)


# In[6]:


data.head(3)


# In[7]:


# dropping uselss columns & too many nulls
data.drop('Unnamed: 0',axis=1,inplace=True)
data.drop('device_type',axis=1,inplace=True)


# In[8]:


## just making a Checkpoint
data_bp1 = data.copy()


# In[9]:


#Importing Datetime time modelue and creating new column for days of week and Month
import datetime
data.page_vw_ts = pd.to_datetime(data.page_vw_ts)
data['date'] = pd.to_datetime(data.page_vw_ts)


# # How to import plotly
# ## open anaconda prompt & type (conda install -c plotly plotly=4.7.1)

# # above output what is cn & what we can analyze from the same.

# In[10]:


## setting index of dataframe to date for analysis
data.set_index('date',inplace=True)


# In[51]:


data['2019'].shape
data['2018'].shape
data.isnull().sum()


# In[12]:


plt.figure(figsize=(15,8))
sns.countplot(y=data['intgrtd_mngmt_name'])
plt.title('intgrtd_mngmt_name',size=20)


# In[13]:


a = data.pivot_table(['ip_addr'],('ctry_name'),aggfunc = 'count')
a = a.sort_values('ip_addr',ascending=False).head(20)


# In[14]:


plt.figure(figsize=(10,5))
sns.barplot(a.ip_addr,a.index)
plt.title("County Which Most Traffic is from ?",size=20)


# ### We see the most of the ip's are from USA,India & Japan

# In[15]:


plt.figure(figsize=(10,5))
data.operating_sys.value_counts().head(20).plot.barh(color = ('green'))
plt.title('Which is Most used OS ?',size=15)


# In[16]:


data.page_url.value_counts().head(10).plot.barh(color='red')
plt.title('Which Website is Visited Most from most frequent 10?',size=15)


# In[17]:


useless = ['city','st','sec_lvl_domn','operating_sys','wk','mth','yr']
data.drop(useless,axis=1,inplace=True)


# In[18]:


data.dropna(inplace=True)


# In[19]:


data_bp2 = data.copy()
data.head(3)


# In[20]:


data['dayofmonth'] = data.index.day
data['dayofweek'] = data.index.dayofweek
data['date'] = data.page_vw_ts.dt.date
data['time'] = data.page_vw_ts.dt.time


# ### Creating a new data set only for Vizualization using only Indias data

# In[21]:


Sdata = data[data.ctry_name == 'India']


# In[23]:


Sdata.head(2)


# In[24]:


#Making Pair plot for Sdata
plt.figure(figsize=(10,10))
sns.heatmap(Sdata.corr(),annot=True,square=True)


# In[25]:


sns.scatterplot(y='VISIT',x='ENGD_VISIT',data=Sdata)


# In[26]:


# Making a Scatter plot for important Features


# In[27]:


plt.scatter(Sdata.VIEWS,Sdata.dayofweek,color='green')
plt.scatter(Sdata.VIEWS,Sdata.ENGD_VISIT,color='red')
plt.scatter(Sdata.VIEWS,Sdata.VISIT,color='blue')


# In[28]:


#lets see the count plot for important features with days
Sdata.pivot_table(['VISIT','VIEWS','ENGD_VISIT'],('dayofweek')).plot.bar()


# In[29]:


#Some Preprocessing
ttt= ['date','ip_addr']
ttt = data[ttt]
ttt= ttt.set_index('date')
ttt = ttt.reset_index()
data['cn'] = 1
te = data.pivot_table(['date'],('ip_addr'),aggfunc=lambda x: len(x.unique()))


# In[30]:


# Lets start with Modelling Part
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler


# In[31]:


#Arranging data according to Distinct Ip adress for all Features
temp = data.pivot_table(['cn','VIEWS','VISIT','ENGD_VISIT'],('ip_addr'),aggfunc='sum')


# In[32]:


temp[temp.index == 'ffffa1ae0a36534162e15078911536c8914a7a7f90c20f267fb4c62a40eec34d']


# In[33]:


#TT = ['VISIT','ENGD_VISIT','day']
df = pd.DataFrame(temp)
df.dropna(inplace=True) ## Dropoing if any null value remaining


# In[34]:


df['indays'] = te['date']


# In[35]:


df


# In[37]:


sns.pairplot(df)


# In[38]:


#Creating Cluster using Kmeans Algorithm.
## using Kmean to make 2 cluster group 
km = KMeans(n_clusters=2)
y_pred = km.fit_predict(df)
df['cluster'] = y_pred


# In[39]:


sns.scatterplot(df.VISIT,df.ENGD_VISIT,hue=df.cluster,)
plt.scatter(km.cluster_centers_[:,1],km.cluster_centers_[:,0],color='red')


# In[41]:


import plotly.express as px


# In[42]:


px.scatter(df,x='VISIT',y='ENGD_VISIT',color='cluster'
           ,hover_data=df)


# In[43]:


temp11 = df.sample(1000,random_state=10)


# In[44]:


fig = px.scatter_3d(temp11, x='ENGD_VISIT', y='VIEWS', z='VISIT',
                    color='cluster',opacity=0.8,hover_data=df)
fig.show()


# In[45]:


df.cluster.value_counts()


# In[46]:


df[df.cluster ==1]


# In[47]:


df[(df.cluster == 0) & (df.ENGD_VISIT > 0)]


# ## The Cluster has Detected the IP has Bot so can say that the cluster has good accracy

# In[ ]:




