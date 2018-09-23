
# coding: utf-8

# Use breast cancer data to create a model to predict type of breast cancer. Also determine which traits indicate whether or not an individual will be diagnosed.
# 
# Data Source: https://www.kaggle.com/uciml/breast-cancer-wisconsin-data

# In[93]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


# In[94]:


def normalize(data):
    x = (data-data.min())/(data.max()-data.min())
    return x

def transform(data):
    x,lam = stats.boxcox(data)
    return x

def visual(df,plot_type):
    plt.rcParams["figure.figsize"] = (23,13)
    fig,ax = plt.subplots(nrows=5,ncols=6)
    i = 0
    j = 0
    
    if plot_type == "hist":
        for col in df.columns:
            if col == "Price":
                ax[i,j].hist(df[col],alpha=0.5,label=col,color="orange")
                ax[i,j].set_title(col + " Distribution")
            else:
                ax[i,j].hist(df[col],alpha=0.5,label=col)
                ax[i,j].set_title(col + " Distribution")
            if j == 5:
                i += 1
                j = 0
            else:
                j += 1
    if plot_type == "box":
        for col in df.columns:
            ax[i,j].boxplot(df[col])
            ax[i,j].set_title(col)
            if j == 5:
                i += 1
                j = 0
            else:
                j += 1
    if plot_type == "qq":
        norm = np.random.normal(0, 1, df.shape[0])
        norm.sort()
        for col in df.columns:
            sorted_col = list(df[col])
            sorted_col.sort()
            if col == "Price":
                ax[i,j].plot(norm,sorted_col,color="orange")
                ax[i,j].set_title(col + " Distribution")
            else:
                ax[i,j].plot(norm,sorted_col)
                ax[i,j].set_title(col + " Distribution")
            
            z = np.polyfit(norm,df[col], 1)
            p = np.poly1d(z)
            

    fig.tight_layout()
    plt.show()


# In[95]:


data = pd.read_csv("data.csv")
data.head(3)


# In[96]:


data.shape


# In[97]:


data["diagnosis"].unique()


# In[98]:


data["diagnosis"] = pd.factorize(data["diagnosis"])[0]


# In[99]:


data["diagnosis"].unique()


# In[100]:


data.isnull().sum()


# In[101]:


data = data.drop('Unnamed: 32',1)
data = data.drop('id',1)


# ## Visualizing Data Distributions and Outliers

# In[102]:


data_features = data.loc[:,(data.columns!="diagnosis")]
data_features = data_features.apply(normalize)


# In[103]:


visual(data_features,"hist")


# Distributions appear slightly skewed in most of the independent features. May need to transform the data to get normal distributions to get a better fit to our models.

# In[104]:


# check for outliers
visual(data_features,"box")


# There appear to be some outliers in the data, lets try to remove some of the siginificant outliers to improve our model's accuracy.

# In[105]:


data_features.shape


# In[106]:


data.shape


# In[107]:


z = np.abs(stats.zscore(data_features))
z1 = np.abs(stats.zscore(data))


# In[108]:


data_features = data_features[(z<3).all(axis=1)]
data = data[(z1<3).all(axis=1)]


# In[109]:


data_features.shape


# In[110]:


data.shape


# In[111]:


visual(data_features,"box")


# In[112]:


data_features = data_features.replace(0,0.000001)


# In[113]:


for col in data_features.columns:
    data_features[col] = transform(data_features[col])


# In[114]:


visual(data_features,"hist")


# Data distributions look a lot better, but we did have to sacrifice a portion of our data. This may impact our results later on. Furthermore, our dataset is rather small, so we may not need to take a sample of our dataset to fit our models. 
