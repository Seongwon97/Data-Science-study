import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
import sklearn.linear_model as lm

df = pd.read_csv('C:/python_file/bmi_data_lab3.csv')
df_temp=df.copy()
df_nonNa=df.dropna(how='any')
height_temp=np.array(df_temp ['Height (Inches)'])
weight_temp=np.array(df_temp ['Weight (Pounds)'])
height=np.array(df_nonNa ['Height (Inches)'])
weight=np.array(df_nonNa ['Weight (Pounds)'])

height_na=[]
weight_na=[]
for i in range(len(df)):
    if np.isnan(height_temp[i]) :
        weight_na.append(weight_temp[i])
    if np.isnan(weight_temp[i]) :
        height_na.append(height_temp[i])

arr_height_na=np.array(height_na)
arr_weight_na=np.array(weight_na)

E1 =lm.LinearRegression()
E1.fit(height[:, np.newaxis], weight)
E2 =lm.LinearRegression()
E2.fit(weight[:, np.newaxis], height)

pxh = np.array([height.min()-1, height.max()+1]) 
pyw = E1.predict(pxh[:, np.newaxis]) 
pxw = np.array([weight.min()-1, weight.max()+1])
pyh = E2.predict(pxw[:, np.newaxis])

x=E2.predict( arr_weight_na[ : ,np.newaxis])
y=E1.predict( arr_height_na[ : ,np.newaxis])
fx=np.append( arr_height_na, x)
fy=np.append( y, arr_weight_na)

plt.scatter(height, weight)
plt.scatter(fx, fy, color='r')
plt.plot(pxh, pyw, color='black')
plt.plot(pyh, pxw, color='black')
plt.xlabel('Height (Inches)')
plt.ylabel('Weight (Pounds)')
plt.show()
