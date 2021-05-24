import pandas as pd
import numpy as np
import io
import warnings
warnings.filterwarnings(action='ignore')
import sklearn.linear_model as lm
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold

#Read file
df=pd.read_csv('C:/python_file/linear_regression_data.csv',encoding='utf-8')



#Split the dataset into 4/5 for training and 1/5 for testing.
testset=df.sample(frac=0.2)
trainset=df.drop(testset.index)
training_distance=trainset['Distance']
training_time=trainset['Delivery Time']
test_distance=testset['Distance']
test_time=testset['Delivery Time']

arr_training_distance=np.array(training_distance)
arr_training_time=np.array(training_time)
arr_test_distance=np.array(test_distance)
arr_test_time=np.array(test_time)


#Creating a model using training data
reg=lm.LinearRegression()
reg.fit(arr_training_distance[:, np.newaxis], arr_training_time)


#predict delivery time using test diatance value
x=reg.predict(arr_test_distance[ : ,np.newaxis])

#Create dataframe for output using test data
df_test=pd.DataFrame(arr_test_distance)
df_test.rename(columns={df_test.columns[0]:'Distance'},inplace=True)
df_test["Delivery Time"]=arr_test_time
df_test["Prediction Delivery Time"]=x

print(df_test)



print("\n\nK-Flod")
#enumerate splits
count=1
kfold=KFold(n_splits=5, shuffle=True, random_state=0)
for train,test in kfold.split(df):
    df_train=df.iloc[train]
    df_train=df_train['Distance']
    df_train_t=df.iloc[train]
    df_train_t=df_train_t['Delivery Time']
    df_test=df.iloc[test]
    df_test=df_test['Distance']
    df_test_t=df.iloc[test]
    df_test_t=df_test_t['Delivery Time']

    reg.fit(df_train[:, np.newaxis], df_train_t)


    param_grid={'fit_intercept':['True','False'],'normalize':['True','False'],}
    gscv=GridSearchCV(reg,param_grid,cv=kfold)
    gscv.fit(df_train[:, np.newaxis],df_train_t)
    prediction = gscv.predict(df_test[:,np.newaxis])


    result=pd.DataFrame(df_test)
    result.rename(columns={result.columns[0]:'Distance'},inplace=True)
    result["Delivery Time"]=df_test_t
    result["Prediction Delivery Time"]=prediction
    print("------------------Result of sample test {0}------------------".format(count))
    count+=1
    print(result)
    print('Best parameter: ',gscv.best_params_)
    print('Best score: ',gscv.best_score_)
    print()
