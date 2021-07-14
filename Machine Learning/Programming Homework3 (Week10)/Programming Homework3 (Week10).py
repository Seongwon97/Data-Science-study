import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import DBSCAN
import warnings

warnings.filterwarnings(action='ignore')

df = pd.read_csv('C:/python_file/datasets_lab2_phw3/mushrooms.csv', encoding='utf-8')

print("Dataset")
print(df)

# data preprocessing
print("Internal value by column")
for i in df.columns:
    print(i, ':', df[i].unique())


# data cleaning
df.replace('?', np.nan, inplace=True)
df.fillna(axis=0, method='ffill', inplace=True)

# convert categorical attribute into numeric one
df = df.apply(LabelEncoder().fit_transform)

y = df['class']
x = df.drop(['class'], axis=1)


def clustering(scaler, data_x):
    best_param = {}
    best_purity = 0
    count = 1

    for e in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]:
        for m in [5, 10, 20, 50, 100]:
            for a in ['ball_tree', 'kd_tree', 'brute']:

                params = {'eps': e, 'min_samples': m, 'algorithm': a, 'metric': 'euclidean', 'p': 2}
                dbscan = DBSCAN(**params)

                df['y_pred'] = dbscan.fit_predict(data_x)

                print(count, params)
                print('label :', df['y_pred'].unique())

                sum = 0

                for i in (df['y_pred'].unique()):
                    df_0 = df[(df['y_pred'] == i) & (df['class'] == 0)]
                    df_1 = df[(df['y_pred'] == i) & (df['class'] == 1)]
                    max_num = max(len(df_0), len(df_1))
                    sum += max_num

                purity = sum / len(df)

                print("Purity :", purity)
                print()
                if purity > best_purity:
                    best_param = params
                    best_purity = purity
                count += 1

    print("----------------------------Result : {}----------------------------".format(scaler))
    print("Best case")
    print("Best Parameter :", best_param)
    print("Best Purity :", best_purity)
    print("\n\n")


print("============================Without Scaler============================")
clustering("Without Scaler", x)

print("============================MinMaxScaler============================")
# Scaling is performed using Minmaxscaling.
minmaxScaler = preprocessing.MinMaxScaler()
m_scaled_x = minmaxScaler.fit_transform(x)
clustering("MinMaxScaler", m_scaled_x)

print("============================StandardScaler============================")
standardScaler = preprocessing.StandardScaler()
s_scaled_x = standardScaler.fit_transform(x)
clustering("StandardScaler", s_scaled_x)
