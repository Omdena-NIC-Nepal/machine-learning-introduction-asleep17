
#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# In[12]:


# Uplaoding Data
df = pd.read_csv("../Data/BostonHousing.csv") 
df.head()


# In[13]:


# Filling empty data if there are some
print(df.isnull().sum())
df.fillna(df.mean(), inplace=True)


# <span>Removing the outlier</span>

# In[23]:


# List of columns to remove outliers from
cols = ['lstat', 'tax', 'rm']

# Loop through each column and remove outliers using IQR method
for col in cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

# Show remaining shape of the data
print("Data shape after outlier removal:", df.shape)


# In[24]:


#Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df.drop('medv', axis=1))

df_scaled = pd.DataFrame(X_scaled, columns=df.columns[:-1])
df_scaled['medv'] = df['medv'].values



# <span>Train / Test split</span>

# In[26]:


X = df_scaled.drop('medv', axis=1)
y = df_scaled['medv']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)


# In[30]:


# exporting the processed data
df.to_csv("../Data/processed_boston.csv", index=False)


# In[ ]:




