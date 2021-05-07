import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt

A = [30,40,50,60,40]
B = [200,300,800,600,300]
C = [10,20,20,20,20]
D = [4, 4, 1, 2, 5]


data= np.array([A,B,C,D])
print("population covariance matrix (N)")
covMatrix=np.cov(data,bias=True)
print(covMatrix)


print("\nsample covariance matrix (N-1)")
covMatrix=np.cov(data,bias=False)
print(covMatrix)
