import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import preprocessing as pr
import matplotlib.pyplot as plt
import sklearn
from pandas import DataFrame as df
from sklearn import linear_model as lm
import math
df1=df(data={'Gender':['Female','Male'],'Age':[21,35],'Height':[110,120],'Weight':[45,11],'BMI':[0,1]})
df1=pd.read_csv('C:/python_file/bmi_data_lab3.csv')

df2 = df1.copy()
df_female = df1.loc[df2['Sex']=='Female'].copy()
df_male = df1.loc[df2['Sex']=='Male'].copy()

Height = np.array(df_male['Height (Inches)']) 
Weight = np.array(df_male['Weight (Pounds)'])

weight_nan = []
height_nan = []

for i in range(0, len(Height)):
    if np.isnan(Height[i]):
        weight_nan.append(Weight[i])
    if np.isnan(Weight[i]) :
        height_nan.append(Height[i])

df1.dropna(inplace=True)
df3=df1.copy()
df2 = df1.loc[df3['Sex']=='Male'].copy()
height = np.array(df2['Height (Inches)']) 
weight = np.array(df2['Weight (Pounds)'])

E = lm.LinearRegression()#
E.fit(height[:, np.newaxis], weight)#

E2 = lm.LinearRegression()#
E2.fit(weight[:, np.newaxis], height)#

x_nan = np.array(height_nan)
y_nan = np.array(weight_nan)



x = E2.predict(y_nan[:, np.newaxis])
y = E.predict(x_nan[:, np.newaxis])


predic_x = np.append(x_nan, x)
predic_y = np.append(y, y_nan)

px_height = np.array([height.min()-1, height.max()+1]) 
py_weight = E.predict(px_height[:, np.newaxis]) 

px_weight = np.array([weight.min()-1, weight.max()+1])
py_height = E2.predict(px_weight[:, np.newaxis])

plt.scatter(height, weight)
plt.scatter(predic_x, predic_y, color='r')
plt.plot(px_height, py_weight, color='black')
plt.plot(py_height, px_weight, color='black')
plt.title('Male Dataset')
plt.xlabel('Height (Inches)')
plt.ylabel('Weight (Pounds)')
plt.show()


Height = np.array(df_female['Height (Inches)']) 
Weight = np.array(df_female['Weight (Pounds)'])

weight_nan = []
height_nan = []

for i in range(0, len(Height)):
    if np.isnan(Height[i]):
        weight_nan.append(Weight[i])
    if np.isnan(Weight[i]):
        height_nan.append(Height[i])

df1.dropna(inplace=True)
df3=df1.copy()
df2 = df1.loc[df3['Sex']=='Female'].copy()
height = np.array(df2['Height (Inches)']) 
weight = np.array(df2['Weight (Pounds)'])

E = lm.LinearRegression()
E.fit(height[:, np.newaxis], weight)

E2 = lm.LinearRegression()
E2.fit(weight[:, np.newaxis], height)

x_nan = np.array(height_nan)
y_nan = np.array(weight_nan)
x = E2.predict(y_nan[:, np.newaxis])
y = E.predict(x_nan[:, np.newaxis])


predic_x = np.append(x_nan, x)
predic_y = np.append(y, y_nan)

px_height = np.array([height.min()-1, height.max()+1]) 
py_weight = E.predict(px_height[:, np.newaxis]) 

px_weight = np.array([weight.min()-1, weight.max()+1])
py_height = E2.predict(px_weight[:, np.newaxis])

plt.scatter(height, weight)
plt.scatter(predic_x, predic_y, color='r')
plt.plot(px_height, py_weight, color='black')
plt.plot(py_height, px_weight, color='black')
plt.title('Female Dataset')
plt.xlabel('Height (Inches)')
plt.ylabel('Weight (Pounds)')
plt.show()
