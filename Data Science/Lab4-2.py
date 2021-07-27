import pandas as pd
import numpy as np
import io
import warnings
warnings.filterwarnings(action='ignore')
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold


#Read file
df=pd.read_csv('C:/python_file/decision_tree_data.csv',encoding='utf-8')


#convert data using labelEncoder
labelEncoder = LabelEncoder()
labelEncoder.fit(df['level'])
df['level'] = labelEncoder.transform(df['level'])
labelEncoder.fit(df['lang'])
df['lang'] = labelEncoder.transform(df['lang'])
labelEncoder.fit(df['tweets'])
df['tweets'] = labelEncoder.transform(df['tweets'])
labelEncoder.fit(df['phd'])
df['phd'] = labelEncoder.transform(df['phd'])



#make DecisionTree Model
tree_model= DecisionTreeClassifier()
count=1
kfold=KFold(n_splits=10, shuffle=True, random_state=0)
for train,test in kfold.split(df):
    df_train=df.iloc[train]
    train_x=np.array(df_train.drop(['interview'],1))
    train_y=np.array(df_train['interview'])
    
    df_test=df.iloc[test]
    test_x=np.array(df_test.drop(['interview'],1))
    test_y=np.array(df_test['interview'])


    param_grid={'max_depth' : np.arange(1,10)}
    gscv=GridSearchCV(tree_model, param_grid, cv=kfold)
    #create model
    gscv.fit(train_x,train_y)
    gscv.predict(test_x)

    result_test=df.iloc[test]
    #predict using test_data
    result_test['Predict']=gscv.predict(test_x)

    print("\n------------------Result of sample test {0}------------------".format(count))
    count+=1
    print(result_test)
    print('predict best parameters: ',gscv.best_params_)
    print('predict est score: ',gscv.best_score_)
