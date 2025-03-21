#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis
# - Analyzing the data (finding patterns inside data,outliers)

# ## Major EDA Operations:
#  1)EDA is used for understanding the dataset
#  
# 2)JOB - Indexing & Filtering inside the data
# 
#  3)Find the Statistical Analysis of Data
#  
#  4)Identify the NAN/NULL vals
#  
#  5)Identify the outliers
# 
#  6)Finding the Duplicate vals

# ## Data Cleaning : Steps

# ### Step-1 fixing data inconsistency
# 
# ### Step-2 Cleaning of null vals
# 
# ### step-3 deleting the duplicate vals
# 
# ### Step-4 Trimming the outliers(removing)

# # Data Cleaning

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#loading the dataset

df = pd.read_csv("C:/Users/Chandrika/Documents/codegnan/Data Analysis/census+income/adult.data") #(csv dataset)
print(df)


# In[3]:


#display in proper format
from IPython.display import display
display(df.head())


# ### Step-1 fixing data inconsistency
# 

# In[4]:


cols = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'martial-status', 'occupation', 'relationship', 
        'race', 'sex', 'captial-gain', 'captial-loss', 'hours-per-week', 'native-country', 'income']


# In[5]:


#assigning col names 

df = pd.read_csv("C:/Users/Chandrika/Documents/codegnan/Data Analysis/census+income/adult.data", names = cols)
display(df)


# # EDA-1 : Understanding the dataset

# In[6]:


df.head()


# In[7]:


df.tail()


# In[8]:


#random sample of 10 records
df.sample(10)


# ### Features with datatype of int64, object
# 

# In[9]:


df.select_dtypes(include = 'int64').columns


# In[10]:


df.select_dtypes(include = 'object').columns


# In[11]:


df.info() #together numerical + object


# ### Pairplot

# In[12]:


sns.pairplot(df)
#can del features if there is a relation b/w them


# In[13]:


df.shape


# In[14]:


df.describe() #numerical vals


# In[15]:


df.describe(include = ['object']) #cateogrical vals


# In[16]:


df['martial-status']


# In[17]:


df['martial-status'].unique()  #unique vals in martial status col


# In[18]:


df.income.unique()   #unique vals in income col


# In[19]:


df.workclass.unique()
#spaces infront of vals and ? in data are prblm here
#unique vals in workclass col


# In[20]:


df.relationship.unique()


# In[21]:


df.race.unique()


# ## DC-1 : fixing the inconsistency

# In[22]:


df['workclass'] = df.workclass.str.strip()  #updating the col


# In[23]:


df.workclass.unique()
#spaces are removed 


# In[24]:


#replacing ? with nan val

df['workclass'] = df.workclass.replace('?',np.nan)


# In[25]:


df.workclass.unique()


# ## EDA-2: Indexing and Filtering the data

# In[26]:


df.loc[1:6, 'age':'education']   #1 to 6, age - educ


# In[27]:


df.iloc[1:7, 0:4]   # age to edu col using iloc


# In[28]:


df.iloc[1:7, [0,3]]  #only age and educ col


# ## Filtering 
# ### sorting data

# In[29]:


df.sort_values('age', ascending = True)


# In[30]:


df.nlargest(4,'age')


# In[31]:


df.nsmallest(4,'age')


# In[32]:


df[df.age < 20]


# In[33]:


df[ (df.age > 20) & (df.age < 30) ]


# ### filter: applying the masking

# In[34]:


(df.age < 20) | (df.age > 70)  #false vals display here 


# In[35]:


df['age'][(df.age < 20) | (df.age > 70)]  #true vals are shown here
#displaying only one col age is masked by cond 


# In[36]:


df['age'][(df.age < 20) | (df.age > 70)].to_frame() #for displaying with col name


# In[37]:


print(type(df['age'][(df.age < 20) | (df.age > 70)]  ))


# In[38]:


#dtype - to see vals inside of data


# In[39]:


sns.countplot(x = 'income', hue = 'income', data = df[df.age < 30])
#counting ppl < 30, categorized based on income
#hue - different colors


# In[40]:


sns.countplot(x = 'workclass', hue = 'workclass', data = df[df.age < 30])


# In[41]:


sns.countplot(x = 'workclass', hue = 'workclass', data = df[df.age < 30])

plt.xticks(rotation = 45) #to overcome overlapping x lables
plt.show()


# # EDA-3 : Statistical Analysis

# In[42]:


# total time all the employees are working
df['hours-per-week'].sum()


# In[43]:


# avg ppl are working
df['hours-per-week'].mean()


# In[44]:


# some ppl work long hours some ppl work for few hours
df['hours-per-week'].median()


# In[45]:


# who is working min hours per week
df['hours-per-week'].min()


# In[46]:


# max hour per week
df['hours-per-week'].max()


# In[47]:


# who is having highest frequenncy working hours
df['hours-per-week'].mode()


# In[48]:


df['hours-per-week'].count()


# In[49]:


# sal < 50k and sal >50k
df.groupby('income').count()


# In[50]:


df['hours-per-week'].std()


# In[51]:


# Cov() - shows the relation b/w vars (Covariance - how 2 cols relate eachother)

df[['age', 'hours-per-week']].cov()

#same job but diff in output only 
# large + ve val : pos cov (age inc numb of hrs inc)
# - ve val : neg cov
# zero : no cov

# in corrrelation val will be bounded b/w -1 and +1(near to 0 no rel, near to -1 have high -ve corel, +1 +ve corel)  
# in covariance -infinte to +inifinte val (cant find out exact rel)


# its easy to predict the correlation rather than covariance

# In[52]:


df['age'].corr(df['hours-per-week'])


# - ML models r baised(give more favour) for the large vals,
# then take correl rather than cov 

# # Heatmap

# used for visualization of the rel b/w vars

# In[53]:


#another way to find corr
corr_matrix = df[['age', 'hours-per-week', 'education-num']].corr()


# In[54]:


print(corr_matrix)


# In[55]:


sns.heatmap(corr_matrix, annot = True, cmap = "coolwarm", fmt = ".2f")


# # EDA-4 : identifying the NaN / Null vals

# In[56]:


df.isnull()


# In[57]:


df.isnull().sum()


# # DC-2 : cleaning the Null vals

# they r many methods 
# 
# Approaches :
# 1) del null record / column
# 
# 2) fill the const vals in place "null"
# 
# 3) fill the mean, median & mode
# 
# 4) use bfill / ffill methods
# 
# 5) use interpolate method
# 
# 6) use supervised learning model - in ML

# # Approach-1 : Del null vals

# In[58]:


df1 = df.copy()
df1.dropna(inplace = True) # updating dataframe here itself
#del the nan vals records


# In[59]:


df1.isnull().sum()


# In[60]:


df1.shape


# In[61]:


df.shape


# In[62]:


df2 = df.copy()


# In[63]:


# deleting complete record with null vals
df2.dropna(axis = 1, inplace = True)


# In[64]:


df2.isnull().sum()


# In[65]:


df2.shape


# In[66]:


df.shape


# # Approach : 2 const vals

# In[67]:


df3 = df.copy()
#creating own null vals
df3.loc[(df3.age >= 30) & (df3.age <= 40), 'age'] = np.nan #assigning nan vals


# In[68]:


df3.isna().sum()  #isnull==isna


# In[69]:


df3 = df3.age.fillna('160') #filling null vals
# (or) df3['age'] = df3.age.fillna('160')


# In[70]:


df.isna().sum() 


# # Approach-3 : Fill mean, median & mode

# In[71]:


df4 = df.copy()


# In[72]:


#creating dummy null vals

df4.loc[(df4.age >= 30) & (df4.age <= 40), 'age'] = np.nan
df4.isnull().sum()


# ### Replacing null val with the mean

# In[73]:


#cal mean val of age
mean_val = df4.age.mean()
mean_val


# In[74]:


df4['age'] = df4.age.fillna(mean_val)


# In[75]:


#null vals in age are replaced with the mean val
df4.isnull().sum()


# ### Replacing the null vals with the median
# 
# 1) data is skewed
# 2) data contains outliers

# In[76]:


df5 = df.copy()


# In[77]:


df5.isnull().sum()


# In[78]:


df5.loc[(df5.age >= 30) & (df5.age <= 40), 'age'] = np.nan


# In[79]:


df5.isnull().sum()


# In[80]:


median_val = df5.age.median()


# In[81]:


median_val


# In[82]:


df5['age'] = df5.age.fillna(median_val)


# In[83]:


df5.isnull().sum()


# # Replacing the null vals with mode

# In[84]:


df6 = df.copy()


# In[85]:


df6.isnull().sum()


# In[86]:


mode_val = df6.workclass.mode()


# In[87]:


mode_val
print(type(mode_val))  #its a series datatype so we have to pass as a str


# In[88]:


df6['workclass'] = df6.workclass.fillna("Private")   #we have to pass as a string instead of mode_val


# In[89]:


df6.isnull().sum()


# # Approach 4.Use bfill / ffill 
when we have cluster data we can use bfill / ffill 
# In[90]:


df_7 = df.copy()
df_7.isnull().sum()


# In[91]:


df_7 = df_7.ffill()
#taking forward val and filling null vals


# In[92]:


df_7.isnull().sum()


# In[93]:


df7 = df.copy()


# In[94]:


df7.isnull().sum()


# In[95]:


df7 = df7.bfill()


# In[96]:


df7.isnull().sum()


# # Approach 5.Using the interpolate Method
when we have continous series data
it is used to fill null vals based upon the patterns inside the dataset
# In[97]:


df8 = df.copy()


# In[98]:


df8.loc[(df8.age >= 30)& (df8.age <= 40), 'age'] = np.nan


# In[99]:


df8.isnull().sum()


# In[100]:


df8['age'] = df8.age.interpolate(method = 'linear')
#interpolate & ffill can't handle the first null value


# In[101]:


df8.isnull().sum()
#this 1 might be first null val
#can fill this null val with help of bfill or const vall


# In[102]:


#finding first null value
df8.head()


# In[103]:


df_8 = df.copy()


# In[104]:


df_8.loc[(df_8.age >= 30) & (df_8.age <= 40), 'age'] = np.nan


# In[105]:


df_8.isnull().sum()


# In[106]:


df_8['age'] = df_8.age.interpolate(method = 'polynomial', order = 2)


# In[107]:


df_8.isnull().sum()


# In[108]:


df_8.head()


# In[109]:


df7 = df.copy()


# In[110]:


df7.loc[(df7.age >= 30) & (df7.age <= 40), 'age'] = np.nan


# In[111]:


df7.isnull().sum()


# In[112]:


df7['age'] = df7.age.interpolate(method = 'spline', order = 2)


# In[113]:


df7.isnull().sum()


# In[114]:


df7.head()


# In[115]:


df_0 = df.copy()


# In[116]:


df_0['age'] = df_0.age.interpolate(method = 'time', order = 2)


# In[117]:


df_0.isnull().sum()


# # EDA-5: Identifying the Duplicates

# In[118]:


#finding duplicate elements
df.duplicated().sum()  #rows


# In[119]:


df[df.duplicated()]   #duplicated records defualt - first


# # DC-3 : Deleting duplicate Rows 

# In[120]:


df.shape


# In[121]:


df_1 = df.copy()


# In[122]:


df_1.drop_duplicates(inplace = True)  #last elements are deleted bcz defualt is first


# In[123]:


df_1.shape


# In[124]:


# Displaying both records
df[df.duplicated(keep = False)]


# In[125]:


# Return the duplicate records after keeping the first record
df[df.duplicated(keep = 'first')]  #shows second records


# In[126]:


# Return the duplicate records after keeping the last record
df[df.duplicated(keep = 'last')]   #returns first record

#deleting the first duplicate and keeping the last duplicate
# may be we want to keep store last info about the customer

df_1.drop_duplicated(inplace = True, keep = "last")
# # EDA-6 : Identifying the Outliers 
Methods : IQR, Boxplot, Z-Score, Clipping
# Outlier : extreme values that deviate significantly from rest of the day

# ## 1) IQR (Interquartile Range) & 2)Boxplot
IQR measures the spread of the middle 50% of the data & also identify the outliersQ1 (First Quantile, 25%)
Q3 (Third Quantile, 75%)

IQR = Q3 - Q1

lower bound (Q1) = Q1 - 1.5 * IQR
upper bound (Q3) = Q3 + 1.5 * IQR


- difference b/w percentage and percentile?
80% marks in cls
percentage - 80%
percentile - 100

-Quartile is a special type percentile (Q1 = 25% percentile, Q3 = 75% percentile, Q2 = 50% percentile)
# In[127]:


sns.boxplot(x = "age", data = df)


# In[128]:


Q1 = df.age.quantile(.25)


# In[129]:


Q1


# In[130]:


Q3 = df.age.quantile(.75)


# In[131]:


Q3


# In[132]:


IQR = Q3 - Q1


# In[133]:


IQR


# In[134]:


lower_bound = Q1 - 1.5 * IQR


# In[135]:


lower_bound


# In[136]:


upper_bound = Q3 + 1.5 * IQR


# In[137]:


upper_bound

Finding the outliers
# In[138]:


outlier = df['age'][(df.age < lower_bound) | (df.age > upper_bound)]  # masking


# In[139]:


outlier


# # DC-4 : Deleting the Outliers

# In[140]:


df10 = df.copy()


# In[141]:


df10 = df[(df['age'] >= lower_bound) & (df['age'] <= upper_bound)] 


# In[142]:


df10.shape


# In[143]:


df.shape


# In[144]:


df.columns


# In[145]:


sns.boxplot(x = 'sex', y = 'age', data = df)


# In[147]:


sns.boxplot(x = 'sex',y = 'age', hue = 'income', data =df)


# # Removing Outliers using
# # 2) Z-Score - (data - mean) / SD
# applicable on numerical vals

# In[148]:


df22 = df.copy()


# In[149]:


df22.isnull().sum()


# In[181]:


df.age.mean()


# In[182]:


df.age.std()


# In[183]:


df.age


# In[152]:


#creating a new col inside dataset named z-score
df22['z_score'] = (df['age'] - df.age.mean()) / df['age'].std()


# In[153]:


df22


# In[166]:


outliers = df22['z_score'][(df22.z_score < -3) | (df22.z_score > 3)]


# In[167]:


outliers


# In[168]:


outliers.sum()


# In[169]:


df11 = df.copy()


# In[177]:


#remove the outliers
df11 = df22[(df22['z_score'] >= -3 ) & (df22['z_score'] <= 3)]


# In[178]:


df11


# In[179]:


df11.shape


# In[180]:


df.shape


# # Removing the Outliers Using
# ## 4) Clipping

# In[184]:


df12 = df.copy()


# In[185]:


df12['age'] = df['age'].clip(lower = 20, upper = 80) #removing data form <20 & >80


# In[186]:


df12.age.unique()


# In[ ]:




