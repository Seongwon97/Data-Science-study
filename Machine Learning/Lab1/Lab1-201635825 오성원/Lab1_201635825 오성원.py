import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, cross_val_score
import seaborn as sns

col_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'value']
df = pd.read_csv('C:/python_file/ML_lab1_datasets/car_mini.csv', header=None, names=col_names)


#preprocessing
#Change to Category Value
labelEncoder = LabelEncoder()
df['buying'] = labelEncoder.fit_transform(df['buying'])
df['maint'] = labelEncoder.fit_transform(df['maint'])
df['doors'] = labelEncoder.fit_transform(df['doors'])
df['persons'] = labelEncoder.fit_transform(df['persons'])
df['lug_boot'] = labelEncoder.fit_transform(df['lug_boot'])
df['safety'] = labelEncoder.fit_transform(df['safety'])

df[['value']] = df[['value']].replace('good', 'acc')
df[['value']] = df[['value']].replace('vgood', 'acc')
df['value'] = labelEncoder.fit_transform(df['value'])

df[['doors']] = df[['doors']].replace('5more', 5)
df[['persons']] = df[['persons']].replace('more', 6)



X = df.drop('value', axis=1)
y = df['value']


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

kf = KFold(n_splits=10)


#SVM
from sklearn.svm import SVC
svclassifier = SVC()
parameters = {'C': [0.1, 1.0, 10.0], 'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'gamma': [0.01, 0.1, 1.0, 10.0]}

clf = GridSearchCV(svclassifier, parameters, cv=kf)
clf.fit(X_train, y_train)
clf.predict(X_test)
score = cross_val_score(clf, X, y, cv=kf)
print("SVM")
print("Best parameters:", clf.best_params_)
print("Score:", score)
print("Best score:", clf.best_score_)
print("Average score:", score.mean())

y_pred = clf.predict(X)
print("Accuracy score %s" %accuracy_score(y, y_pred))
#Get the confusion matrix
cf_matrix = confusion_matrix(y, y_pred)
print("Confusion Matrix\n", cf_matrix)
sns.heatmap(cf_matrix, annot=True)
plt.title("SVM Confusion Matrix")
plt.show()




#Logistic Regression
from sklearn.linear_model import LogisticRegression
logisticRegr = LogisticRegression()
parameters = {'C': [0.1, 1.0, 10.0], 'solver': ['liblinear', 'lbfgs', 'sag'],
              'max_iter': [50, 100, 200]}

clf = GridSearchCV(logisticRegr, parameters, cv=kf)
clf.fit(X_train, y_train)
clf.predict(X_test)
scores=cross_val_score(clf, X, y, cv=kf)
print("\n\n\nLogistic Regression")
print("Best parameters:", clf.best_params_)
print("Score:", score)
print("Best score:", clf.best_score_)
print("Average score:", score.mean())

y_pred = clf.predict(X)
print("Accuracy score %s" %accuracy_score(y, y_pred))
#Get the confusion matrix
cf_matrix = confusion_matrix(y, y_pred)
print("Confusion Matrix\n", cf_matrix)
sns.heatmap(cf_matrix, annot=True)
plt.title("Logistic Regression Confusion Matrix")
plt.show()




#Random forest
from sklearn.ensemble import RandomForestClassifier
randomforest = RandomForestClassifier()
parameters = {'criterion': ['gini', 'entropy'], 'n_estimators': [1, 10, 100],
              'max_depth': [1, 10, 100]}

clf = GridSearchCV(randomforest, parameters, cv=kf)
clf.fit(X_train, y_train)
clf.predict(X_test)
print("\n\n\nRandom Forest")
print("Best parameters:", clf.best_params_)
print("Score:", score)
print("Best score:", clf.best_score_)
print("Average score:", score.mean())

y_pred = clf.predict(X)
print("Accuracy score %s" %accuracy_score(y, y_pred))
#Get the confusion matrix
cf_matrix = confusion_matrix(y, y_pred)
print("Confusion Matrix\n", cf_matrix)
sns.heatmap(cf_matrix, annot=True)
plt.title("Random forest Confusion Matrix")
plt.show()


