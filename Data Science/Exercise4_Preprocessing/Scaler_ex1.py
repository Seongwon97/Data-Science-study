import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
import sklearn.linear_model as lm

df = pd.read_excel('C:/python_file/bmi_data_phw3.xlsx','dataset', index_col=None)

print(df)

for i in df:
    print("Feature name:", i,", type:",df[i].dtypes)

#Plot height histograms (bins=10) for each BMI value
for j in range(0,5):
    data=[]
    for i in range(len(df)):
        if(df['BMI'][i].item() == j):
            if df['Height (Inches)'][i] < 65:
                data.append(5)
            elif 65 <= df['Height (Inches)'][i] < 66:
                data.append(15)
            elif 66 <= df['Height (Inches)'][i] < 67:
                data.append(25)
            elif 67 <= df['Height (Inches)'][i] < 68:
                data.append(35)
            elif 68 <= df['Height (Inches)'][i] < 69:
                data.append(45)
            elif 69 <= df['Height (Inches)'][i] < 70:
                data.append(55)
            elif 70 <= df['Height (Inches)'][i] < 71:
                data.append(65)
            elif 71 <= df['Height (Inches)'][i] < 72:
                data.append(75)
            elif 72 <= df['Height (Inches)'][i] < 73:
                data.append(85)
            elif 73 <= df['Height (Inches)'][i] :
                data.append(95)
    if data:
        name = ['~65', '65~66', '66~67', '67~68','67~68','68~69','70~71','71~72','72~73','73~']
        plt.hist(data, bins=[0,10,20,30,40,50,60,70,80,90,100], rwidth=0.7)
        plt.xticks([5,15,25,35,45,55,65,75,85,95], name, fontsize=6)
        if j==0:
            plt.title('Height histogram-Bmi Extremely weak')
        elif j==1:
            plt.title('Height histogram-Bmi Weak')
        elif j==2:
            plt.title('Height histogram-Bmi Normal')
        elif j==3:
            plt.title('Height histogram-Bmi Overweight')
        elif j==4:
            plt.title('Height histogram-Bmi Obesity')
        plt.xlabel("Height(Inches)")
        plt.ylabel("Number of students")
        plt.show()
#----------------------------------------------------------------------------------------
#Plot weight histograms (bins=10) for each BMI value
for j in range(0,5):
    data=[]
    for i in range(len(df)):
        if(df['BMI'][i].item() == j):
            if df['Weight (Pounds)'][i] < 85:
                data.append(5)
            elif 85 <= df['Weight (Pounds)'][i] < 95:
                data.append(15)
            elif 95 <= df['Weight (Pounds)'][i] < 105:
                data.append(25)
            elif 105 <= df['Weight (Pounds)'][i] < 115:
                data.append(35)
            elif 115 <= df['Weight (Pounds)'][i] < 125:
                data.append(45)
            elif 125 <= df['Weight (Pounds)'][i] < 135:
                data.append(55)
            elif 135 <= df['Weight (Pounds)'][i] < 145:
                data.append(65)
            elif 145 <= df['Weight (Pounds)'][i] < 155:
                data.append(75)
            elif 155 <= df['Weight (Pounds)'][i] < 165:
                data.append(85)
            elif 165 <= df['Weight (Pounds)'][i]:
                data.append(95)
    if data:
        name = ['~85', '85~95', '95~105', '105~115','115~125','125~135','135~145','145~155','155~165','165~']
        plt.hist(data, bins=[0,10,20,30,40,50,60,70,80,90,100], rwidth=0.7)
        plt.xticks([5,15,25,35,45,55,65,75,85,95], name, fontsize=6)
        if j==0:
            plt.title('Weight histogram-Bmi Extremely weak')
        elif j==1:
            plt.title('Weight histogram-Bmi Weak')
        elif j==2:
            plt.title('Weight histogram-Bmi Normal')
        elif j==3:
            plt.title('Weight histogram-Bmi Overweight')
        elif j==4:
            plt.title('Weight histogram-Bmi Obesity')
        plt.xlabel("Weight(Pounds)")
        plt.ylabel("Number of students")
        plt.show()
        
#-----------------------------------------------------------------------------------------------------
#Plot Scaling Result(Standard Scaler)
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
#Plot Scaling Result(MinMax Scaler)
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
#Plot Scaling Result(Robust Scaler)
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

