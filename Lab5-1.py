import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import seaborn as sn
import warnings
warnings.filterwarnings(action='ignore')

#Read the test datafile and save it in iris.
iris=pd.read_csv('C:/python_file/Iris_test_dataset.csv', encoding='utf-8')
target=iris['Species']


#Read 10 training datasets and save them in the samples list
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

"""
The bagging function is a function
that returns the predicted values by receiving the training data and test data.
"""
def bagging(train_data, test_data):
    #Create test and train data.
    predict=[]
    train=train_data.copy()
    test=test_data.copy()
    test_x=test_data
    test_x.drop(['Species'], axis=1, inplace=True)

    
    for i in range(10):
        train_y=train[i]['Species']
        train_x=train[i]
        train_x.drop(['Species'], axis=1, inplace=True)
        #Create model by inserting train data into the DecisionTreeCalssifier.
        tree_model=tree.DecisionTreeClassifier().fit(train_x, train_y)

        #Predict results using model
        result=tree_model.predict(test_x)

        #Insert the generated result into the predict list.
        predict.append(result)

    return predict


#Obtain the prediction through the bagging function.
prediction=bagging(samples,iris)

predicted=[]

for i in range(len(iris)):
    #variables for counting for each flower
    se=0
    ve=0
    vi=0
    max_vote=-1
    max_index=''

    #Conduct a test data length and voting each data.
    for j in range(10):
        if(prediction[j][i]=='Iris-setosa'):
            se+=1
        elif(prediction[j][i]=='Iris-versicolor'):
            ve+=1
        elif(prediction[j][i]=='Iris-virginica'):
            vi+=1
        if(max_vote < se):
            max_vote=se
            max_index='Iris-setosa'
        if(max_vote < ve):
            max_vote=ve
            max_index='Iris-versicolor'
        if(max_vote < vi):
            max_vote=ve
            max_index='Iris-virginica'

    predicted.append(max_index)

#Construct a confusion matrix using actual data and predicted result data.
actual_data=np.array(target)       
data={'y_Predicted':predicted, 'y_Actual': actual_data}
df2=pd.DataFrame(data, columns=['y_Actual', 'y_Predicted'])

confusion_matrix=pd.crosstab(df2['y_Actual'], df2['y_Predicted'], rownames=['Actual'], colnames=['Predicted'],margins=True)

print('---------------------------Confusion Matrix---------------------------')
print(confusion_matrix)

print()
print('Accuracy_score')
print(accuracy_score(predicted, actual_data))


