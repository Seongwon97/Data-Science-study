import pandas as pd
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
import numpy as np
import warnings
warnings.filterwarnings(action='ignore')





df = pd.read_csv('C:/python_file/insurance.csv')

labelEncoder = LabelEncoder()
df['sex'] = labelEncoder.fit_transform(df['sex'])
df['smoker'] = labelEncoder.fit_transform(df['smoker'])
df['region'] = labelEncoder.fit_transform(df['region'])
print(df)
print("\n==============================================================")

x5 = df.drop('charges', axis=1)
y = df['charges'].values.reshape(-1, 1)

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2,  1, 5, 10, 20]}

#Linear regression========================================================================================

linear = LinearRegression()
score = cross_val_score(linear, x5, y, scoring='neg_mean_squared_error', cv=10)

mean_score = np.mean(score)
print("\nLinear Regression")
print("Mean score:", mean_score)

print("\n==============================================================")

#Lasso regression========================================================================================
lasso = Lasso()

lasso_regressor = GridSearchCV(lasso, parameters, scoring='neg_mean_squared_error', cv=10)

lasso_regressor.fit(x5, y)

print("\nLasso Regression")
print("Best Parameters:", lasso_regressor.best_params_)
print("Best Score:", lasso_regressor.best_score_)


print("\n==============================================================")

#Ridge regression========================================================================================

ridge = Ridge()
ridge_regressor = GridSearchCV(ridge, parameters, scoring='neg_mean_squared_error', cv=10)

ridge_regressor.fit(x5, y)

print("\nRidge Regression")
print("Best Parameters:", ridge_regressor.best_params_)
print("Best Score:", ridge_regressor.best_score_)


print("\n==============================================================")

#Elastic Net regression========================================================================================

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2,  1, 5, 10, 20],
              'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]}

elastic = ElasticNet()
elastic_regressor = GridSearchCV(elastic, parameters, scoring='neg_mean_squared_error', cv=10)

elastic_regressor.fit(x5, y)

print("\nElastic Net Regression")
print("Best Parameters:", elastic_regressor.best_params_)
print("Best Score:", elastic_regressor.best_score_)

