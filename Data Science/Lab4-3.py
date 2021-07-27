import pandas as pd
import numpy as np
import io
import math
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings(action='ignore')


#Read file
df=pd.read_csv('C:/python_file/knn_data.csv',encoding='utf-8')





count=1
#enumerate splits
kfold=KFold(n_splits=5, shuffle=True, random_state=0)
for train,test in kfold.split(df):
    df_train=df.iloc[train]
    df_train=df_train[['longitude','latitude']]
    df_train_lang=df.iloc[train]
    df_train_lang=df_train_lang['lang']
    df_test=df.iloc[test]
    df_test=df_test[['longitude','latitude']]
    df_test_lang=df.iloc[test]
    df_test_lang=df_test_lang['lang']



    #create KNN model
    knn = KNeighborsClassifier()
    param_grid = {'n_neighbors': np.arange(1,25)}
    knn_gscv = GridSearchCV (knn, param_grid , cv=5)
    #create model
    knn_gscv.fit(df_train, df_train_lang)
    
    result_test=df.iloc[test]
    #predict using test_data
    result_test['Predicted lang']=knn_gscv.predict(df_test)

    print("------------------Result of sample test {0}------------------".format(count))
    count+=1
    print(result_test)
    print("Predict bet params: {0}".format(knn_gscv.best_params_))
    print("Predict best score: {0}\n".format(knn_gscv.best_score_))

    




