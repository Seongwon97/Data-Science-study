import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing

df = pd.read_excel('C:/python_file/bmi_data_phw3.xlsx','dataset', index_col=None)

#Create an array of heights and weights
#that store the height and weight values.
height=np.array(df["Height (Inches)"])
weight=np.array(df["Weight (Pounds)"])

E=lm.LinearRegression()
E.fit(height[:,np.newaxis],weight)

#For each data, w 'is calculated using E
#and height values, which are regression data.
ex=np.array(height)
ey=E.predict(ex[:,np.newaxis]) #variable ey is w'
e=weight-ey

#Normalize the value of e.
mean_e=sum(e)/len(e)
std_e=np.std(e)
ze=np.array(e-mean_e)/std_e

#Set the bin size to 10
plt.hist(ze, bins=10, rwidth=0.8)
plt.xlabel("Ze")
plt.ylabel("Frequency")
plt.show()


a= np.arange(0, 2, 0.01)
max_num=-1
max_a=-1
#Find the value of a that best matches when predicted from 0 to 1
for j in a:
    df2=df.copy()
    match=0
    #Decide a value α (≥0); for records with z e <-α , set BMI = 0;
    #for those with z e >α , set BMI = 4
    for i in range(len(df2)):
        if ze[i]> j:
            df2.at[i,'BMI']=4
        elif ze[i]< (-j):
            df2.at[i,'BMI']=0
        else:
            df2.at[i,'BMI']=-1
        if df.at[i,'BMI']==df2.at[i,'BMI']:
            match+=1

    if max_num<=match:
        max_num=match
        max_a=j

print("When α is {0}, the predicted value is most consistent.".format(round(max_a,2)))
