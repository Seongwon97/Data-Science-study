import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt

#Read the file
df = pd.read_excel('C:/python_file/HW10_dataset1.xlsx')

#print all column name
print("Attribute name")
for i in range(len(df.columns)):
    print("{0} . {1}".format(i+1, df.columns[i]))

#Select the target column
target_column=int(input("\nSelect the target: "))
value_max_iter=int(input("Enter the Max_iter (Initial:600): "))
print("---------------------------------------------------------------------------------------------------")
target=df.columns[target_column-1]

#It is divided into input data and target data.
X=np.array(df.drop([target], 1).astype(float))
Y=np.array(df[target])

#Normalize using MinMaxScaler
scaler=preprocessing.MinMaxScaler()
X_scaled=scaler.fit_transform(X)


#A function that outputs data belonging to each cluster.
def print_cluster(data, kmeans, k):
    cluster_row=[[],[],[]]
    for i in range(len(kmeans.labels_)):
        if kmeans.labels_[i]==0:
            cluster_row[0].append(i)
        elif kmeans.labels_[i]==1:
            cluster_row[1].append(i)
        elif kmeans.labels_[i]==2:
            cluster_row[2].append(i)

    for i in range(k):
        print("Cluster",i+1)
        cluster=df.loc[cluster_row[i], : ]
        print(cluster,"\n")
    
#A function that calculate prediction
def calculate_prediction(kmeans,X,Y):
    correct=0
    for i in range(len(X)):
        predict_me=np.array(X[i].astype(float))
        predict_me=predict_me.reshape(-1,len(predict_me))
        prediction=kmeans.predict(predict_me)
        if prediction[0]==Y[i]:
            correct+=1
    print("Prediction: ",correct/len(X))


kmeans=KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
              n_clusters=2,n_init=10, n_jobs=1, precompute_distances='auto',
              random_state=None, tol=0.0001,verbose=0)
kmeans.fit(X_scaled)
print("Result in max_iter=600")
print_cluster(df, kmeans, 2)
if target_column==4:
    calculate_prediction(kmeans, X,Y)
if target_column==5:
    calculate_prediction(kmeans, X,Y)


print("---------------------------------------------------------------------------------------------------")
kmeans=KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=value_max_iter,
              n_clusters=2,n_init=10, n_jobs=1, precompute_distances='auto',
              random_state=None, tol=0.0001,verbose=0)
kmeans.fit(X_scaled)
print("Result in max_iter=",value_max_iter)
print_cluster(df, kmeans, 2)



