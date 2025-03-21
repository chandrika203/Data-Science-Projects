#!/usr/bin/env python
# coding: utf-8

# # Feature Engineering

# ## 1) Feature Transformation
# - converting categorical to numerical or large vals into particular range

# ### 1.1 Feature Encoding

# ### 1.1.1 Label Encoding
Example - 1
# In[78]:


from sklearn.preprocessing import LabelEncoder


# In[79]:


import pandas as pd


# In[80]:


data = {'Color' : ['Red', 'Yellow', 'Green']}


# converting data into dataframe bcz this lib works only on dataframe

# In[81]:


df = pd.DataFrame(data)
#print(df)

encoder = LabelEncoder()  #creating an encoder object for Label Encoder function

encoder.fit(df['Color'])  

df['Color_encoded'] = encoder.transform(df['Color'])   #created new column

df


# Example - 2 

# In[82]:


df = pd.read_csv("C:/Users/Chandrika/Documents/codegnan/Data Analysis/census+income/adult.data")


# In[83]:


df


# In[84]:


cols = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'martial-status', 'occupation', 'relationship', 
        'race', 'sex', 'captial-gain', 'captial-loss', 'hours-per-week', 'native-country', 'income']


# In[85]:


df = pd.read_csv("C:/Users/Chandrika/Documents/codegnan/Data Analysis/census+income/adult.data", names = cols)
display(df)


# In[86]:


df.workclass.unique()


# In[87]:


df['workclass'] = df.workclass.str.strip()


# In[88]:


import numpy as np


# In[89]:


df['workclass'] = df.workclass.replace('?', np.nan)


# In[90]:


df.workclass.unique()

TASK - 1 create a new col with int values assigned to workclass categories
# In[91]:


data = df['workclass']
data_1 = pd.DataFrame(data)
data_1


# In[92]:


encoder = LabelEncoder()  #creating an encoder object for Label Encoder function

encoder.fit(data_1['workclass'])  

df['workclass'] = encoder.transform(data_1['workclass'])   #created new column

df


# In[93]:


# to check the sequence

encoder.classes_


# ### 1.1.2 Ordinal Encoding

# Example - 1

# In[94]:


from sklearn.preprocessing import OrdinalEncoder


# In[95]:


dataset = [['Phd'], ['UG'], ['MS'], ['UG'], ['Phd']]


# In[96]:


encoder = OrdinalEncoder(categories = [['UG', 'MS', 'Phd']])

encoded_data = encoder.fit_transform(dataset)

encoded_data


# Example - 2

# In[97]:


df


# In[98]:


df.education.unique()


# In[99]:


df['education'] = df.education.str.strip()


# In[100]:


df.education.unique()


# In[101]:


education_categories = ['Preschool', '1st-4th', '5th-6th', '7th-8th', '9th', '10th', '11th', '12th',
                       'Prof-school', 'Some-college','HS-grad', 'Bachelors', 'Masters', 'Doctorate', 'Assoc-acdm', 
                       'Assoc-voc']

encoder = OrdinalEncoder(categories = [education_categories])
df['education'] = encoder.fit_transform(df[['education']])

df


# In[102]:


df.education.unique()


# ### 1.1.3 One Hot Encoding

# Example - 1

# In[103]:


data = {'Country' : ['India', 'USA', 'Canada', 'India', 'USA']}

df_new = pd.DataFrame(data)

df_encoded = pd.get_dummies(df_new, columns = ['Country'])  #convert into binary 

df_encoded

# 1 - True , 0 - False


# using sklearn lib for one hot encoding

# In[104]:


from sklearn.preprocessing import OneHotEncoder


# In[105]:


# creating object

obj = OneHotEncoder(sparse = False, dtype = int) #sparse is cretaing 1 col but we want to 2 cols so creating new_col


# In[106]:


new_col = obj.fit_transform(df[['sex']]) #2d array


# In[107]:


print(type(new_col)) #return 2 arrays we need dataframe
 
# 2 cols are created 1-Male, 2-Female


# In[108]:


#converting numpy array into dataframe
#giving meaning full names to columns(created Dataframe cols) using func get_feature_names_out

encoded_df = pd.DataFrame(new_col, columns = obj.get_feature_names_out(['sex']))

encoded_df


# In[109]:


#concatenating encode_df into df

df = pd.concat([df, encoded_df], axis = 1)

df


# In[110]:


df = df.drop('sex', axis =1)


# In[111]:


df


# ## 1.2 Feature Scaling

# ### 1.2.1 Min-Max Scaling (normalization technique)

# In[112]:


from sklearn.preprocessing import MinMaxScaler

for particular range use feature_range
 
obj = MinMaxScaler(feature_range = (-1, 1))
# In[113]:


obj = MinMaxScaler()

df['fnlwgt'] = obj.fit_transform(df[['fnlwgt']]) #should pass 2d only


# In[114]:


df


# ### 1.2.2 Standardization

# In[115]:


from sklearn.preprocessing import StandardScaler


# In[116]:


obj = StandardScaler()

df['hours-per-week'] = obj.fit_transform(df[['hours-per-week']])


# In[117]:


df


# ### 1.2.3 Robust Scaler

# In[118]:


from sklearn.preprocessing import RobustScaler


# In[119]:


obj = RobustScaler()

df['age'] = obj.fit_transform(df[['age']])


# In[120]:


df


# Converting all the remaining columns from categorical into numerical

# steps:
#     1) create a list of columns
#     
#     2) for loop

# In[121]:


encoder = LabelEncoder()

list_1 = ['martial-status', 'occupation', 'relationship', 'race', 'native-country', 'income']

for col_name in list_1:
    df[col_name] = encoder.fit_transform(df[col_name])

df


# Feature Scaling on all columns

# In[122]:


obj = MinMaxScaler()


# In[123]:


cols = ['workclass', 'education', 'education-num', 'martial-status', 'occupation', 'relationship', 
        'race', 'captial-gain', 'captial-loss',  'native-country', 'income']


# In[124]:


for col_name in cols:
    df[col_name] = obj.fit_transform(df[[col_name]])
df


# # 2)Feature Selection

# ## 2.1 Correlation Matrix

# In[125]:


corr = df.corr()


# In[126]:


corr


# through visualization 

# In[127]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[128]:


plt.figure(figsize = (20,10))
sns.heatmap(corr, annot = True, cmap = 'coolwarm', fmt = '.2f')
plt.show()


# if your dataset is big, have more features(columns) below code is used

# In[129]:


high_corr = corr.unstack()
high_corr


# 256 combinations are produced with same relation 

# In[130]:


high_corr = corr.unstack().reset_index()
high_corr


# assigning meaning full names to columns (level 0...)

# In[131]:


high_corr.columns = ['Feature1', 'Feature2', 'Correlation']
high_corr


# showing correlation > 30% and == 1 is same so eliminating the columns

# In[132]:


high_corr = high_corr[(high_corr['Correlation'] > 0.3) & (high_corr['Correlation'] != 1)]

high_corr


# sorting correlation values - to wind up similar values

# In[133]:


high_corr = high_corr.sort_values(by = 'Correlation', ascending = True)

high_corr


# ### 2.2 Chi-Square Test

# to check relation b/w two particular vars

# In[134]:


from scipy.stats import chi2_contingency


# creating a contingency matrix require for chi-square method

# In[135]:


contingency_matrix = pd.crosstab(df['education'], df['education-num'])

contingency_matrix


# chi2_contingency() returns 4 values 
# 
# to not recevie values _ is placed bcz we cant skip it

# In[136]:


_ , p_value, _ ,_ = chi2_contingency(contingency_matrix)  #to not recevie values _ is placed bcz we cant skip 


# In[137]:


p_value


# In[138]:


var1 , p_value, var2 ,var3 = chi2_contingency(contingency_matrix)


# In[139]:


var1


# In[140]:


var2


# In[141]:


var3


# In[142]:


p_value


# # 3) Feature Balancing

# ### 3.1 OverSampling
# #### 3.1.2 RandomOverSampling

# 2 categories of ppl are there here
# 
# 1) 0.0 - 76% having < 50k
# 
# 2) 1.0 - 24% ppl having > 50k salary 

# In[143]:


df.income.value_counts().plot(kind = 'pie', autopct = '%0.0f%%')


# In[144]:


get_ipython().system('pip install imbalanced-learn')


# In[145]:


from imblearn.over_sampling import RandomOverSampler, SMOTE


# In[146]:


x = df.drop('income', axis = 1)  
#entire dataset expect income column bcz we deleted and this updated data is stored in x


# In[147]:


y = df.loc[: , ['income']]  
#income column with all the rows stored in y


# not majority - resample all the classes but the majority classes(expect majority class)

# In[148]:


obj_ros = RandomOverSampler(sampling_strategy = 'not majority')

x_ros, y_ros = obj_ros.fit_resample(x,y)


# In[149]:


y.value_counts() #Before 


# In[150]:


y_ros.value_counts() #After resampling


# ### 3.1.1 SMOTE

# 50 % of miniority class is added here

# In[151]:


obj_smote = SMOTE(sampling_strategy = .5)

x_smote, y_smote = obj_smote.fit_resample(x,y)


# In[152]:


y_smote.value_counts()


# ## 3.2 UnderSampling

# In[153]:


from imblearn.under_sampling import RandomUnderSampler


# In[155]:


obj_rus = RandomUnderSampler(sampling_strategy = 'not minority')
x_rus, y_rus = obj_rus.fit_resample(x,y)


# In[156]:


y_rus.value_counts()


# In[157]:


from imblearn.under_sampling import EditedNearestNeighbours


# In[158]:


obj_enn = EditedNearestNeighbours(n_neighbors = 5)
x_enn, y_enn = obj_enn.fit_resample(x,y)


# In[159]:


y_enn.value_counts()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




