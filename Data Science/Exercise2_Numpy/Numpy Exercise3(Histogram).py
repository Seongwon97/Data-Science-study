import numpy as np
import matplotlib.pyplot as plt

weight = np.random.random(100)*50+40
height = np.random.random(100)*60+140
bmi = weight / (height*height*0.0001)
data=[]

for i in range(100):
    if bmi[i] < 18.5:
        data.append(5)
    elif 18.5 <= bmi[i] < 25:
        data.append(15)
    elif 25 <= bmi[i] < 30:
        data.append(25)
    elif 30 <= bmi[i]:
        data.append(35)
        
name = ['underweight', 'healthy', 'overweight', 'obese']
plt.hist(data, bins=[0,10,20,30,40], rwidth=0.8)
plt.xticks([5,15,25,35], name)
plt.title('BMI for 100 student')
plt.xlabel("BMI status")
plt.ylabel("Number of students")
plt.show()
