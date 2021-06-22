import pandas as pd
import numpy as np
from sklearn import tree
import warnings
warnings.filterwarnings(action='ignore')
"""

"""

iris=pd.read_csv('C:/python_file/Iris_test_dataset.csv', encoding='utf-8')
labels=iris['Species']


samples=[]
sample=pd.read_csv('C:/python_file/Iris_train_datasets/Iris_bagging_dataset (1).csv', encoding='utf-8')
samples.append(sample)
sample=pd.read_csv('C:/python_file/Iris_train_datasets/Iris_bagging_dataset (2).csv', encoding='utf-8')
samples.append(sample)
sample=pd.read_csv('C:/python_file/Iris_train_datasets/Iris_bagging_dataset (3).csv', encoding='utf-8')
samples.append(sample)
sample=pd.read_csv('C:/python_file/Iris_train_datasets/Iris_bagging_dataset (4).csv', encoding='utf-8')
samples.append(sample)
sample=pd.read_csv('C:/python_file/Iris_train_datasets/Iris_bagging_dataset (5).csv', encoding='utf-8')
samples.append(sample)
sample=pd.read_csv('C:/python_file/Iris_train_datasets/Iris_bagging_dataset (6).csv', encoding='utf-8')
samples.append(sample)
sample=pd.read_csv('C:/python_file/Iris_train_datasets/Iris_bagging_dataset (7).csv', encoding='utf-8')
samples.append(sample)
sample=pd.read_csv('C:/python_file/Iris_train_datasets/Iris_bagging_dataset (8).csv', encoding='utf-8')
samples.append(sample)
sample=pd.read_csv('C:/python_file/Iris_train_datasets/Iris_bagging_dataset (9).csv', encoding='utf-8')
samples.append(sample)
sample=pd.read_csv('C:/python_file/Iris_train_datasets/Iris_bagging_dataset (10).csv', encoding='utf-8')
samples.append(sample)

def bagging(sample,iteration, sample_size, label):
    y=sample[label]
    X=sample
    X.drop([label],axis=1,inplace=True)
    model=tree.DecisionTreeClassifier().fit(X,y)
    return model


iteration=10
sample_size=30
label='Species'
models=[]

for sample in samples:
    model=bagging(sample,iteration,sample_size, label)
    models.append(model)

x_test=iris[label]
X_test=iris
X_test.drop([label],axis=1,inplace=True)

predictions=[]
for model in models:
    p=model.predict(X_test)
    predictions.append(p)


compare = iris
for i in range(iteration):
    compare[i]=predictions[i]

print(compare.tail())


voting=[]
class_cnt=[0,0,0]

for col in range(len(train)):
    for row in range(iteration):
        if(row<iteration):
            index=predictions[row][col]
            if(index=='Iris-setosa'): class_cnt[0]+=1
            elif(index=='Iris-versicolor'): class_cnt[1]+=1
            else: class_cnt[2]+=1
    max_label_index=class_cnt.index(max(class_cnt))
    if(max_label_index==0): voting.append('Iris-setosa')
    elif(max_label_index==1): voting.append('Iris-versicolor')
    else: voting.append('Iris-virginica')
    class_cnt=[0,0,0]

compare['prediction']=voting
print(compare.head())

for j in range(len(compare)):
    if(compare['labels'][j]==0): compare['labels'][j]=2
    elif(compare['labels'][j]==1): compare['labels'][j]=0
    else: compare['labels'][j]=1
