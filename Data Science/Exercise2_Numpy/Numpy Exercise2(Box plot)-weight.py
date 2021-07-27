import numpy as np
import matplotlib.pyplot as plt

weight = np.random.random(100)*50+40
height = np.random.random(100)*60+140
bmi = weight / (height*height*0.0001)


underweight = []
healthy = []
overweight = []
obese = []

for i in range(100):
    if bmi[i] < 18.5:
        underweight.append(weight[i])
    elif 18.5 <= bmi[i] < 25:
        healthy.append(weight[i])
    elif 25 <= bmi[i] < 30:
        overweight.append(weight[i])
    elif 30 <= bmi[i]:
        obese.append(weight[i])

plotData = [underweight, healthy, overweight, obese]
plt.title('BMI for 100 student')
plt.ylabel('Weight')
plt.boxplot(plotData)
plt.show()
