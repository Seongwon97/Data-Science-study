import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing


df = pd.read_csv('C:/python_file/bmi_data_lab3_1.csv')

print(df,"\n")

print("Feature names\m", df.columns, "\n")

print("Data type : ", type(df))
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


#-----------------------------------------------------------------------------------------------------

for j in range(0,5):
    data=[]
    for i in range(len(df)):
        if(df['BMI'][i].item() == j):
            if df['Weight (Pounds)'][i] < 100:
                data.append(5)
            elif 100 <= df['Weight (Pounds)'][i] < 106:
                data.append(15)
            elif 106 <= df['Weight (Pounds)'][i] < 112:
                data.append(25)
            elif 112 <= df['Weight (Pounds)'][i] < 118:
                data.append(35)
            elif 118 <= df['Weight (Pounds)'][i] < 124:
                data.append(45)
            elif 124 <= df['Weight (Pounds)'][i] < 130:
                data.append(55)
            elif 130 <= df['Weight (Pounds)'][i] < 136:
                data.append(65)
            elif 136 <= df['Weight (Pounds)'][i] < 142:
                data.append(75)
            elif 142 <= df['Weight (Pounds)'][i] < 148:
                data.append(85)
            elif 148 <= df['Weight (Pounds)'][i]:
                data.append(95)
    if data:
        name = ['~100', '100~106', '106~112', '112~118','118~124','124~130','130~136','136~142','142~148','148~']
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
