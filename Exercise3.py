import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt

A=[30,200,10,4]
B=[40,300,20,4]
C=[50,800,20,1]
D=[60,600,20,2]
E=[40,300,20,5]

data= np.array([A,B,C,D,E])

print("\nPopulation data")
covMatrix=np.cov(data,bias=True)
print(covMatrix)

sn.heatmap(covMatrix, annot=True, fmt='g')
plt.show()


print("\nSample data")
covMatrix=np.cov(data,bias=False)
print(covMatrix)


sn.heatmap(covMatrix, annot=True, fmt='g')
plt.show()
