import pandas as pd
import numpy as np
import random
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import seaborn as sn
import warnings
warnings.filterwarnings(action='ignore')

#Read the test datafile and save it in iris.
iris=pd.read_csv('C:/python_file/Iris_test_dataset.csv', encoding='utf-8')
target=iris['Species']


#Read the 10 sample data and put them all together in df.
sample1=pd.read_csv('C:/python_file/Iris_train_datasets/Iris_bagging_dataset (1).csv', encoding='utf-8')
sample2=pd.read_csv('C:/python_file/Iris_train_datasets/Iris_bagging_dataset (2).csv', encoding='utf-8')
df=pd.concat([sample1, sample2],axis=0)
sample=pd.read_csv('C:/python_file/Iris_train_datasets/Iris_bagging_dataset (3).csv', encoding='utf-8')
df=pd.concat([df,sample],axis=0)
sample=pd.read_csv('C:/python_file/Iris_train_datasets/Iris_bagging_dataset (4).csv', encoding='utf-8')
df=pd.concat([df,sample],axis=0)
sample=pd.read_csv('C:/python_file/Iris_train_datasets/Iris_bagging_dataset (5).csv', encoding='utf-8')
df=pd.concat([df,sample],axis=0)
sample=pd.read_csv('C:/python_file/Iris_train_datasets/Iris_bagging_dataset (6).csv', encoding='utf-8')
df=pd.concat([df,sample],axis=0)
sample=pd.read_csv('C:/python_file/Iris_train_datasets/Iris_bagging_dataset (7).csv', encoding='utf-8')
df=pd.concat([df,sample],axis=0)
sample=pd.read_csv('C:/python_file/Iris_train_datasets/Iris_bagging_dataset (8).csv', encoding='utf-8')
df=pd.concat([df,sample],axis=0)
sample=pd.read_csv('C:/python_file/Iris_train_datasets/Iris_bagging_dataset (9).csv', encoding='utf-8')
df=pd.concat([df,sample],axis=0)
sample=pd.read_csv('C:/python_file/Iris_train_datasets/Iris_bagging_dataset (10).csv', encoding='utf-8')
df=pd.concat([df,sample],axis=0)

#Override index of data in df
a=list(range(len(df)))
df.index=a


"""
The bagging function is a function
that returns the predicted values by receiving the training data and test data.
"""
def bagging(train_data, test_data, k):
    #Create k training datasets containing 30 data randomly using bootstrap method.
    sample=[]
    for j in range(k):

        df2 = pd.DataFrame(index=range(0), columns=['Id', 'SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species'])
        for i in range(30):
            ran=random.randrange(300)
            temp=df.loc[ran]
            df2.loc[i]=temp

        sample.append(df2)

    #Create test and train data.
    predict=[]
    train=sample.copy()
    test=test_data.copy()
    test_x=test
    test_x.drop(['Species'], axis=1, inplace=True)
    
    for i in range(k):
        train_y=train[i]['Species']
        train_x=train[i]
        train_x.drop(['Species'], axis=1, inplace=True)
        #Create model by inserting train data into the DecisionTreeCalssifier.
        tree_model=tree.DecisionTreeClassifier(max_depth=1).fit(train_x, train_y)
        
        #Predict results using model
        result=tree_model.predict(test_x)

        #Insert the generated result into the predict list.
        predict.append(result)
        
    return predict

"""
Function to output the confusion matrix
by receiving the K value and releasing the enemble reading.
"""
def ensemble(k):

    #Recall bagging function using k value
    prediction=bagging(df, iris, k)

    predicted=[]
    for i in range(len(iris)):

        #variables for counting for each flower
        se=0
        ve=0
        vi=0
        max_vote=-1
        max_index=''

         #Conduct a test data length and voting each data.
        for j in range(k):
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

    print('=================== K={0} ==================='.format(k))
    print('---------------------------Confusion Matrix---------------------------')
    print(confusion_matrix)
    print()
    print('Accuracy_score')
    print(accuracy_score(predicted, actual_data),'\n')


#Implemented when K is 5,10,20,50,100, respectively.
ensemble(5)
ensemble(10)
ensemble(20)
ensemble(50)
ensemble(100)
