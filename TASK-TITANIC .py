#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


titanic = pd.read_csv('titanic.csv')


# In[4]:


titanic


# In[5]:


titanic.describe()


# In[6]:


sns.heatmap(titanic.corr())
plt.show()


# In[8]:


import matplotlib.pyplot as plt

plt.subplot(1, 2, 1)
strat_train_set["Survived"].hist()
strat_train_set["Pclass"].hist()

plt.subplot(1, 2, 2)
strat_test_set["Survived"].hist()
strat_test_set["Pclass"].hist()

plt.show()


# # Analysing data

# In[9]:


sns.countplot (x = "Survived" , data = titanic)


# In[10]:


#those who are not survived are more than 250 are greater than those who survived nerarly 150


# In[11]:


sns.countplot(x= 'Survived' , hue = 'Sex', data = titanic , palette = 'winter')


# In[12]:


#male survived more than females 


# In[13]:


sns.countplot(x='Survived' , hue = 'Pclass' , data =titanic , palette = 'PuBu')


# In[14]:


## for the analysis we get that the 3rd class people are more survied but also who did not survived are belong to 3rd class more 


# In[15]:


titanic ["Age"].plot.hist()


# In[16]:


# the hightest age group travelling are among the young age bwtween 20-40
# very few passangers in age 70


# In[17]:


titanic ['Fare'].plot.hist(bins=20,figsize = (10,5))


# In[18]:


# we observe that most of the ticket bought are under fare 100 
# and very few are on the higher side of fare  (220-500)


# In[19]:


sns.countplot(x = 'SibSp' , data = titanic , palette ='winter')


# In[20]:


# we observe that most of the passanger do not have their siblings at titanic 


# In[21]:


sns.countplot(x='Parch', data =titanic ,palette= 'rocket')


# In[22]:


# we observe parents are very less who abord


# # Data Wrangling

# In[23]:


titanic.isnull().sum()


# In[24]:


# Cabin has most null values and age too has null values


# In[25]:


sns.boxplot(x='Pclass', y = 'Age' ,data = titanic)


# In[26]:


# We will observe that older age group are travelling more in class 1 


# In[27]:


# we going to drop some coloumn which we are not using


# In[28]:


titanic.drop('Cabin' , axis =1 ,inplace = True )


# In[29]:


titanic.head(2)


# In[30]:


titanic.dropna(inplace = True )


# In[31]:


titanic.isnull().sum()


# In[32]:


# we check that now we don't have any null values


# In[33]:


pd.get_dummies(titanic['Sex']).head()


# In[34]:


sex = pd.get_dummies(titanic['Sex'] , drop_first=True)
sex.head(2)


# In[35]:


embark = pd.get_dummies(titanic['Embarked'])
embark.head(3)


# In[36]:


embark = pd.get_dummies(titanic['Embarked'] , drop_first =True)
embark.head(3)


# In[37]:


pcl = pd.get_dummies(titanic['Pclass'], drop_first = True)


# In[38]:


titanic = pd.concat([titanic, sex, embark, pcl], axis=1)
titanic.head(3)


# In[39]:


titanic.drop(['Name', 'PassengerId', 'Pclass', 'Ticket', 'Sex', 'Embarked'], axis=1, inplace=True)
titanic.head(3)


# In[ ]:




