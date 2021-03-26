import numpy as np
weight = np.random.random(100)*50+40
height = np.random.random(100)*60+140
bmi = weight / (height*height*0.0001)

for i in range(10):
    print("Student{0} bmi is {1}".format( i+1, bmi[i] ))
