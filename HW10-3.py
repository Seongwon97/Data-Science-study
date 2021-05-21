import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

#Read the file
df = pd.read_excel('C:/python_file/HW10_dataset2.xlsx')
df_list = df.values.tolist()

#-----------------------------------------------------
#Single Linkage Hierarchical Clustering
linked = linkage(df, 'single')
dendrogram(linked, orientation='top',
            labels=[1,2,3,4,5],
            distance_sort='descending',
            show_leaf_counts=True)
plt.title("Single Linkage Clustering")
plt.show()

#-----------------------------------------------------
#Complete Linkage Hierarchical Clustering
linked = linkage(df, 'complete')
dendrogram(linked, orientation='top',
            labels=[1,2,3,4,5],
            distance_sort='descending',
            show_leaf_counts=True)
plt.title("Complete Linkage Clustering")
plt.show()

#-----------------------------------------------------
#Average Linkage Hierarchical Clustering
linked = linkage(df, 'average')
dendrogram(linked, orientation='top',
            labels=[1,2,3,4,5],
            distance_sort='descending',
            show_leaf_counts=True)
plt.title("Average Linkage Clustering")
plt.show()


#-----------------------------------------------------
#Centroid Linkage Hierarchical Clustering
linked = linkage(df, 'centroid')
dendrogram(linked, orientation='top',
            labels=[1,2,3,4,5],
            distance_sort='descending',
            show_leaf_counts=True)
plt.title("Centroid Linkage Clustering")
plt.show()


