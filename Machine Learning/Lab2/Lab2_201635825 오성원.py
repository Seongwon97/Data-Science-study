import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('C:/python_file/datasets_lab2_phw3/mouse.csv', encoding='utf-8', names=['x', 'y'])

print(df)

#DBSCAN=================================================
from sklearn.cluster import DBSCAN
for eps in [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]:
    for min_samples in [3, 5, 10, 15, 20, 30, 50, 100]:
        params = {'eps': eps, 'min_samples': min_samples}
        dbscan = DBSCAN(**params)
        dbscan.fit(df)
        df['dbscan_id'] = dbscan.labels_
        
        plt.scatter(df['x'], df['y'], c =df['dbscan_id'], alpha=0.5)

        text = "DBSCAN, eps: " + str(eps) + ",min_samples: ", str(min_samples)
        plt.title(text)
        plt.show()

#k-means=================================================
from sklearn.cluster import KMeans

for n_clusters in [2, 3, 4, 5, 6]:
    for max_iter in [50, 100, 200, 300]:
        params = {'n_clusters': n_clusters, 'max_iter': max_iter}
        k_means = KMeans(**params)
        k_means.fit(df)
        df['k_means_id'] = k_means.labels_
        


        plt.scatter(df['x'], df['y'], c =df['k_means_id'], alpha=0.5)

        text = "K-means, n-clusters: "+str(n_clusters)+",max_iter: "+ str(max_iter)
        plt.title(text)
        plt.show()

#EM clustering==========================================
from sklearn.mixture import GaussianMixture

for n_components in [2, 3, 4, 5, 6]:
    for max_iter in [50, 100, 200, 300]:
        params = {'n_components': n_components, 'max_iter': max_iter}
        gmm = GaussianMixture(**params)
        gmm.fit(df)
        y_predict = gmm.predict(df)
        df['EM_id'] = y_predict


        plt.figure(figsize=(8,8))
        plt.scatter(df['x'], df['y'], c =df['EM_id'], alpha=0.5)
        text = "EM, n_components: "+str(n_components)+",max_iter: "+ str(max_iter)
        plt.title(text)
        plt.show()



