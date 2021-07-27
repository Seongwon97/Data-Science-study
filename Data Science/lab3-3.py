import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
import sklearn.linear_model as lm
import copy


missing_values=[""]
df = pd.read_csv('C:/python_file/bmi_data_lab3.csv', na_values=missing_values)
print(df)

print("#of NaN for each row")
for i in range(len(df)):
    df_temp=df.loc[i]
    print("Index",i, " of NaN: ",df_temp.isna().sum())

print("#of NaN for each column")
print(df.isna().sum())
print("Extract all rows without NaN")
print(df.dropna(how='any'))


df_temp = copy.deepcopy(df)
print("!-- Result of fillna mean --!")
df_temp['Height (Inches)'] = pd.to_numeric(df_temp['Height (Inches)'])
df_temp['Weight (Pounds)'] = pd.to_numeric(df_temp['Weight (Pounds)'])
df_temp['Age'] = pd.to_numeric(df_temp['Age'])
df_temp['BMI'] = pd.to_numeric(df_temp['BMI'])
mean=df_temp.mean()
print(df_temp.fillna(mean))


print("!-- Result of fillna median --!")
print(df.fillna(df.median()))

print("!-- Result of ffill --!")
print(df.fillna(df.fillna(axis=0, method='ffill')))

print("!-- Result of bfill --!")
print(df.fillna(df.fillna(axis=0, method='bfill')))
