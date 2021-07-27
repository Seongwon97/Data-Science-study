import numpy as np
import matplotlib.pyplot as plt

weight = np.random.random(100)*50+40
height = np.random.random(100)*60+140
bmi = weight / (height*height*0.0001)
        
plt.scatter(height, weight, color='r')
plt.xlabel("Height")
plt.ylabel("Weight")
plt.title('Scatter')
plt.show()
