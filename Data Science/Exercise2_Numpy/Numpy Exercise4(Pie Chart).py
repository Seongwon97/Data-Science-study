import numpy as np
import matplotlib.pyplot as plt

weight = np.random.random(100)*50+40
height = np.random.random(100)*60+140
bmi = weight / (height*height*0.0001)

underweight = []
healthy = []
overweight = []
obese = []

for i in bmi:
    if i < 18.5:
        underweight.append(i)
    elif 18.5 <= i < 25:
        healthy.append(i)
    elif 25 <= i < 30:
        overweight.append(i)
    elif 30 <= i:
        obese.append(i)

size=[]
size.append(len(underweight))
size.append(len(healthy))
size.append(len(overweight))
size.append(len(obese))

group = ['Under weight', 'Healthy', 'Over weight', 'Obese']
plt.pie(size, labels = group, autopct = '%.1f%%')
plt.title('BMI for 100 student')
plt.legend(fontsize=6)
plt.show()
