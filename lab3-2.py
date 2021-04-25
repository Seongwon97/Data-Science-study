import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
import sklearn.linear_model as lm


df = pd.read_csv('C:/python_file/bmi_data_lab3_1.csv')

#-----------------------------------------------------------------------------------------------------
df2 = pd.DataFrame(
    {'Height (Inches)' : df['Height (Inches)'],
     'Weight (Pounds)': df['Weight (Pounds)']
     })

scaler = preprocessing.StandardScaler()
scaled_df = scaler.fit_transform(df2)
scaled_df = pd.DataFrame(scaled_df,columns=['Height (Inches)','Weight (Pounds)'])
fig, (ax1, ax2) = plt.subplots(ncols = 2, figsize =(6,5))

ax1.set_title('Before Scaling')
sns.kdeplot(df ['Height (Inches)'], ax=ax1)
sns.kdeplot(df ['Weight (Pounds)'], ax=ax1)
ax2.set_title('After Standard Scaler')
sns.kdeplot(scaled_df ['Height (Inches)'], ax=ax2)
sns.kdeplot(scaled_df ['Weight (Pounds)'], ax=ax2)
plt.show()

#-----------------------------------------------------------------------------------------------------
df2 = pd.DataFrame(
    {'Height (Inches)' : df['Height (Inches)'],
     'Weight (Pounds)': df['Weight (Pounds)']
     })

scaler = preprocessing.MinMaxScaler()
scaled_df = scaler.fit_transform(df2)
scaled_df = pd.DataFrame(scaled_df,columns=['Height (Inches)','Weight (Pounds)'])
fig, (ax1, ax2) = plt.subplots(ncols = 2, figsize =(6,5))

ax1.set_title('Before Scaling')
sns.kdeplot(df ['Height (Inches)'], ax=ax1)
sns.kdeplot(df ['Weight (Pounds)'], ax=ax1)
ax2.set_title('After MinMax Scaler')
sns.kdeplot(scaled_df ['Height (Inches)'], ax=ax2)
sns.kdeplot(scaled_df ['Weight (Pounds)'], ax=ax2)
plt.show()

#-----------------------------------------------------------------------------------------------------
df2 = pd.DataFrame(
    {'Height (Inches)' : df['Height (Inches)'],
     'Weight (Pounds)': df['Weight (Pounds)']
     })

scaler = preprocessing.RobustScaler()
scaled_df = scaler.fit_transform(df2)
scaled_df = pd.DataFrame(scaled_df,columns=['Height (Inches)','Weight (Pounds)'])
fig, (ax1, ax2) = plt.subplots(ncols = 2, figsize =(6,5))

ax1.set_title('Before Scaling')
sns.kdeplot(df ['Height (Inches)'], ax=ax1)
sns.kdeplot(df ['Weight (Pounds)'], ax=ax1)
ax2.set_title('After Robust Scaler')
sns.kdeplot(scaled_df ['Height (Inches)'], ax=ax2)
sns.kdeplot(scaled_df ['Weight (Pounds)'], ax=ax2)
plt.show()

