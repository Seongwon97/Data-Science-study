import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
import sklearn.linear_model as lm

df = pd.read_excel('C:/python_file/bmi_data_phw3.xlsx','dataset', index_col=None)

#Copy the values ​​to the array df_m and df_w for each gender.
df_m=df.loc[df['Sex']=="Male"].copy()
arr=np.array(range(0,len(df_m)))
df_m.index=arr
df_w=df.loc[df['Sex']=="Female"].copy()
arr=np.array(range(0,len(df_w)))
df_w.index=arr


#Linear Regression with male data
height_m=np.array(df_m["Height (Inches)"])
weight_m=np.array(df_m["Weight (Pounds)"])

E_m=lm.LinearRegression()
E_m.fit(height_m[:,np.newaxis],weight_m)

#For each data, w 'is calculated using E
#and height values, which are regression data.
ex_m=np.array(height_m)
ey_m=E_m.predict(ex_m[:,np.newaxis]) #variable ey is w'
e_m=weight_m-ey_m

#Normalize the value of e.
meanE_m=sum(e_m)/len(e_m)
stdE_m=np.std(e_m)
ze_m=np.array(e_m-meanE_m)/stdE_m

#Set the bin size to 10
plt.hist(ze_m, bins=10, rwidth=0.8)
plt.title("Male")
plt.xlabel("Ze")
plt.ylabel("Frequency")
plt.show()

#------------------------------------------------------------------------
#Linear Regression with female data
height_w=np.array(df_w["Height (Inches)"])
weight_w=np.array(df_w["Weight (Pounds)"])

E_w=lm.LinearRegression()
E_w.fit(height_w[:,np.newaxis],weight_w)

#For each data, w 'is calculated using E
#and height values, which are regression data.
ex_w=np.array(height_w)
ey_w=E_w.predict(ex_w[:,np.newaxis]) #variable ey is w'
e_w=weight_w - ey_w

#Normalize the value of e.
meanE_w=sum(e_w)/len(e_w)
stdE_w=np.std(e_w)
ze_w=np.array(e_w-meanE_w)/stdE_w

#Set the bin size to 10
plt.hist(ze_w, bins=10, rwidth=0.8)
plt.title("Female")
plt.xlabel("Ze")
plt.ylabel("Frequency")
plt.show()
#------------------------------------------------------------------------
a= np.arange(0, 2, 0.01)
maxNum_m=-1
maxA_m=-1
#Find the value of a that best matches when predicted from 0 to 1
for j in a:
    df2=df_m.copy()
    match=0
    #Decide a value α (≥0); for records with z e <-α , set BMI = 0;
    #for those with z e >α , set BMI = 4
    for i in range(len(df2)):
        if ze_m[i]> j:
            df2.at[i,'BMI']=4
        elif ze_m[i]< (-j):
            df2.at[i,'BMI']=0
        else:
            df2.at[i,'BMI']=-1
            
        if df_m.at[i,'BMI'] == df2.at[i,'BMI']:
            match+=1

    if maxNum_m<=match:
        maxNum_m=match
        maxA_m=j

#------------------------------------------------------------------------
a= np.arange(0, 2, 0.01)
maxNum_w=-1
maxA_w=-1
#Find the value of a that best matches when predicted from 0 to 1
for j in a:
    df2=df_w.copy()
    match=0
    #Decide a value α (≥0); for records with z e <-α , set BMI = 0;
    #for those with z e >α , set BMI = 4
    for i in range(len(df2)):
        if ze_w[i]> j:
            df2.at[i,'BMI']=4
        elif ze_w[i]< (-j):
            df2.at[i,'BMI']=0
        else:
            df2.at[i,'BMI']=-1
            
        if df_w.at[i,'BMI'] == df2.at[i,'BMI']:
            match+=1

    if maxNum_w<=match:
        maxNum_w=match
        maxA_w=j


print("In male dataset, When α is {0}, the predicted value is most consistent.".format(round(maxA_m,2)))
print("In female dataset, When α is {0}, the predicted value is most consistent.".format(round(maxA_w,2)))

