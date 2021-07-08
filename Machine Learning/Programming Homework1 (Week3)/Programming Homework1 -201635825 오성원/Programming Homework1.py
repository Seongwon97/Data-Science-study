from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import KFold
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
from collections import Counter

df = pd.read_csv('C:/python_file/ML_lab1_datasets/mnist.csv')

print(df)
X = df.drop('label', axis=1)
y = df['label']


from sklearn.model_selection import train_test_split
temp_train, X_test_D1, temp_train, y_test_D1 = train_test_split(X, y, test_size=0.1, shuffle=True, random_state=0)

X_train_D1, temp_test, y_train_D1, temp_test = train_test_split(X, y, test_size=0.9, shuffle=True, random_state=0)

X_train_D2, X_test_D2, y_train_D2, y_test_D2 = train_test_split(X, y, test_size=0.1, shuffle=True, random_state=0)


kf = KFold(n_splits=10)


#=======================================================================================
#Logistic Regression

logisticRegr = LogisticRegression()
parameters = {'C': [0.1, 1.0, 10.0], 
              'solver': ['liblinear', 'lbfgs', 'sag'],
              'max_iter': [50, 100, 200]}

reg_clf = GridSearchCV(logisticRegr, parameters, cv=kf)
reg_clf.fit(X_train_D1, y_train_D1)
print("\n\n\nLogistic Regression")
print("Best parameters:", reg_clf.best_params_)
print("Best score:", reg_clf.best_score_)

y_pred = reg_clf.predict(X_test_D1)
reg_acc = accuracy_score(y_test_D1, y_pred)
print("Accuracy:", reg_acc)

liblinear_result = []
lbfgs_result = []
sag_result = []

print("\nAccuracy according to parameters.")
print("solver-liblinear")
for i in range(len(reg_clf.cv_results_["params"])):
    if reg_clf.cv_results_["params"][i]['solver'] == 'liblinear':
        liblinear_result.append(reg_clf.cv_results_["mean_test_score"][i])
        print("Parameters: ", reg_clf.cv_results_["params"][i], "   Accuracy: ", reg_clf.cv_results_["mean_test_score"][i])

print("solver-lbfgs")
for i in range(len(reg_clf.cv_results_["params"])):
    if reg_clf.cv_results_["params"][i]['solver'] == 'lbfgs':
        lbfgs_result.append(reg_clf.cv_results_["mean_test_score"][i])
        print("Parameters: ", reg_clf.cv_results_["params"][i], "   Accuracy: ", reg_clf.cv_results_["mean_test_score"][i])

print("solver-sag")
for i in range(len(reg_clf.cv_results_["params"])):
    if reg_clf.cv_results_["params"][i]['solver'] == 'sag':
        sag_result.append(reg_clf.cv_results_["mean_test_score"][i])
        print("Parameters: ", reg_clf.cv_results_["params"][i], "   Accuracy: ", reg_clf.cv_results_["mean_test_score"][i])

#Get the confusion matrix
cf_matrix = confusion_matrix(y_test_D1, y_pred)
print("\nLogistic Regression Confusion Matrix\n", cf_matrix)
sns.heatmap(cf_matrix, annot=True, cmap="YlGnBu")
plt.title("Logistic Regression Confusion Matrix")
plt.show()


#3D bar chart
#liblinear
fig = plt.figure(figsize=(5, 5), dpi=150)
ax_l = fig.add_subplot(111, projection='3d')

xlabels = np.array([50, 100, 200])
xpos = np.arange(xlabels.shape[0])

ylabels = np.array([0.1, 1.0, 10.0])
ypos = np.arange(ylabels.shape[0])

xposM, yposM = np.meshgrid(xpos, ypos, copy=False)

liblinear_result = np.array(liblinear_result)
zpos = liblinear_result
zpos = zpos.ravel()

dx = 0.5
dy = 0.5
dz = zpos

ax_l.w_xaxis.set_ticks(xpos + dx/2.)
ax_l.w_xaxis.set_ticklabels(xlabels)

ax_l.w_yaxis.set_ticks(ypos + dy/2.)
ax_l.w_yaxis.set_ticklabels(ylabels)

ax_l.set_xlabel('max_iter')
ax_l.set_ylabel('C')
ax_l.set_zlabel('Accuracy')
ax_l.set_title("Logistic Regression(liblinear)")
values = np.linspace(0.2, 1., xposM.ravel().shape[0])
colors = cm.rainbow(values)
ax_l.bar3d(xposM.ravel(), yposM.ravel(), dz*0, dx, dy, dz, color=colors)
plt.show()


#lbfgs
fig = plt.figure(figsize=(5, 5), dpi=150)
ax_lb = fig.add_subplot(111, projection='3d')
lbfgs_result = np.array(lbfgs_result)
zpos = lbfgs_result
zpos = zpos.ravel()
dz = zpos

ax_lb.w_xaxis.set_ticks(xpos + dx/2.)
ax_lb.w_xaxis.set_ticklabels(xlabels)

ax_lb.w_yaxis.set_ticks(ypos + dy/2.)
ax_lb.w_yaxis.set_ticklabels(ylabels)

ax_lb.set_xlabel('max_iter')
ax_lb.set_ylabel('C')
ax_lb.set_zlabel('Accuracy')
ax_lb.set_title("Logistic Regression(lbfgs)")

ax_lb.bar3d(xposM.ravel(), yposM.ravel(), dz*0, dx, dy, dz, color=colors)
plt.show()

#sag
fig = plt.figure(figsize=(5, 5), dpi=150)
ax_s = fig.add_subplot(111, projection='3d')
sag_result = np.array(sag_result)
zpos = sag_result
zpos = zpos.ravel()
dz = zpos

ax_s.w_xaxis.set_ticks(xpos + dx/2.)
ax_s.w_xaxis.set_ticklabels(xlabels)

ax_s.w_yaxis.set_ticks(ypos + dy/2.)
ax_s.w_yaxis.set_ticklabels(ylabels)

ax_s.set_xlabel('max_iter')
ax_s.set_ylabel('C')
ax_s.set_zlabel('Accuracy')
ax_s.set_title("Logistic Regression(sag)")

ax_s.bar3d(xposM.ravel(), yposM.ravel(), dz*0, dx, dy, dz, color=colors)
plt.show()


#=======================================================================================
#Random forest
randomforest = RandomForestClassifier()
parameters = {'criterion': ['gini', 'entropy'], 
              'n_estimators': [1, 10, 100],
              'max_depth': [1, 10, 100]}

ran_clf = GridSearchCV(randomforest, parameters, cv=kf)
ran_clf.fit(X_train_D1, y_train_D1)
print("\n\n\nRandom Forest")
print("Best parameters:", ran_clf.best_params_)
print("Best score:", ran_clf.best_score_)

y_pred = ran_clf.predict(X_test_D1)
ran_acc = accuracy_score(y_test_D1, y_pred)
print("Accuracy:", ran_acc)

gini_result = []
entropy_result = []

print("\nAccuracy according to parameters.")
print("criterion-gini")
for i in range(len(ran_clf.cv_results_["params"])):
    if ran_clf.cv_results_["params"][i]['criterion'] == 'gini':
        gini_result.append(ran_clf.cv_results_["mean_test_score"][i])
        print("Parameters: ", ran_clf.cv_results_["params"][i], "   Accuracy: ", ran_clf.cv_results_["mean_test_score"][i])

print("criterion-entropy")
for i in range(len(ran_clf.cv_results_["params"])):
    if ran_clf.cv_results_["params"][i]['criterion'] == 'entropy':
        entropy_result.append(ran_clf.cv_results_["mean_test_score"][i])
        print("Parameters: ", ran_clf.cv_results_["params"][i], "   Accuracy: ", ran_clf.cv_results_["mean_test_score"][i])

#Get the confusion matrix
cf_matrix = confusion_matrix(y_test_D1, y_pred)
print("\nRandom Forest Confusion Matrix\n", cf_matrix)
sns.heatmap(cf_matrix, annot=True, cmap="YlGnBu")
plt.title("Random Forest Confusion Matrix")
plt.show()


#3D bar chart
#gini
fig = plt.figure(figsize=(5, 5), dpi=150)
ax_g = fig.add_subplot(111, projection='3d')

xlabels = np.array([1, 10, 100])
xpos = np.arange(xlabels.shape[0])

ylabels = np.array([1, 10, 100])
ypos = np.arange(ylabels.shape[0])

xposM, yposM = np.meshgrid(xpos, ypos, copy=False)

gini_result = np.array(gini_result)
zpos = gini_result
zpos = zpos.ravel()


dx = 0.5
dy = 0.5
dz = zpos

ax_g.w_xaxis.set_ticks(xpos + dx/2.)
ax_g.w_xaxis.set_ticklabels(xlabels)

ax_g.w_yaxis.set_ticks(ypos + dy/2.)
ax_g.w_yaxis.set_ticklabels(ylabels)

ax_g.set_xlabel('n_estimators')
ax_g.set_ylabel('max_depth')
ax_g.set_zlabel('Accuracy')
ax_g.set_title("Random Forest(gini)")
values = np.linspace(0.2, 1., xposM.ravel().shape[0])
colors = cm.rainbow(values)
ax_g.bar3d(xposM.ravel(), yposM.ravel(), dz*0, dx, dy, dz, color=colors)
plt.show()


#entropy
fig = plt.figure(figsize=(5, 5), dpi=150)
ax_en = fig.add_subplot(111, projection='3d')
entropy_result = np.array(entropy_result)
zpos = entropy_result
zpos = zpos.ravel()
dz = zpos

ax_en.w_xaxis.set_ticks(xpos + dx/2.)
ax_en.w_xaxis.set_ticklabels(xlabels)

ax_en.w_yaxis.set_ticks(ypos + dy/2.)
ax_en.w_yaxis.set_ticklabels(ylabels)

ax_en.set_xlabel('n_estimators')
ax_en.set_ylabel('max_depth')
ax_en.set_zlabel('Accuracy')
ax_en.set_title("Random Forest(entropy)")

ax_en.bar3d(xposM.ravel(), yposM.ravel(), dz*0, dx, dy, dz, color=colors)
plt.show()




#=======================================================================================
#SVM
svclassifier = SVC()
parameters = {'C': [0.1, 1.0, 10.0],
              'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
              'gamma': [0.01, 0.1, 1.0, 10.0]}

svm_clf = GridSearchCV(svclassifier, parameters, cv=kf)
svm_clf.fit(X_train_D1, y_train_D1)
#clf.predict(X_test_D1)
print("\n\n\nSVM")
print("Best parameters:", svm_clf.best_params_)
print("Best score:", svm_clf.best_score_)

y_pred = svm_clf.predict(X_test_D1)
svm_acc = accuracy_score(y_test_D1, y_pred)
print("Accuracy:", svm_acc)

linear_result = []
poly_result = []
rbf_result = []
sigmoid_result = []

print("\nAccuracy according to parameters.")
print("kernel-linear")
for i in range(len(svm_clf.cv_results_["params"])):
    if svm_clf.cv_results_["params"][i]['kernel'] == 'linear':
        linear_result.append(svm_clf.cv_results_["mean_test_score"][i])
        print("Parameters: ", svm_clf.cv_results_["params"][i], "   Accuracy: ", svm_clf.cv_results_["mean_test_score"][i])

print("kernel-poly")
for i in range(len(svm_clf.cv_results_["params"])):
    if svm_clf.cv_results_["params"][i]['kernel'] == 'poly':
        poly_result.append(svm_clf.cv_results_["mean_test_score"][i])
        print("Parameters: ", svm_clf.cv_results_["params"][i], "   Accuracy: ", svm_clf.cv_results_["mean_test_score"][i])

print("kernel-rbf")
for i in range(len(svm_clf.cv_results_["params"])):
    if svm_clf.cv_results_["params"][i]['kernel'] == 'rbf':
        rbf_result.append(svm_clf.cv_results_["mean_test_score"][i])
        print("Parameters: ", svm_clf.cv_results_["params"][i], "   Accuracy: ", svm_clf.cv_results_["mean_test_score"][i])

print("kernel-sigmoid")
for i in range(len(svm_clf.cv_results_["params"])):
    if svm_clf.cv_results_["params"][i]['kernel'] == 'sigmoid':
        sigmoid_result.append(svm_clf.cv_results_["mean_test_score"][i])
        print("Parameters: ", svm_clf.cv_results_["params"][i], "   Accuracy: ", svm_clf.cv_results_["mean_test_score"][i])

#Get the confusion matrix
cf_matrix = confusion_matrix(y_test_D1, y_pred)
print("\nSVM Confusion Matrix\n", cf_matrix)
sns.heatmap(cf_matrix, annot=True, cmap="YlGnBu")
plt.title("SVM Confusion Matrix")
plt.show()

#3D bar chart
#linear
fig = plt.figure(figsize=(5, 5), dpi=150)
ax_l = fig.add_subplot(111, projection='3d')

xlabels = np.array([0.01, 0.1, 1.0, 10.0])
xpos = np.arange(xlabels.shape[0])

ylabels = np.array([0.1, 1.0, 10.0])
ypos = np.arange(ylabels.shape[0])

xposM, yposM = np.meshgrid(xpos, ypos, copy=False)

linear_result = np.array(linear_result)
zpos = linear_result
zpos = zpos.ravel()

dx = 0.5
dy = 0.5
dz = zpos

ax_l.w_xaxis.set_ticks(xpos + dx/2.)
ax_l.w_xaxis.set_ticklabels(xlabels)

ax_l.w_yaxis.set_ticks(ypos + dy/2.)
ax_l.w_yaxis.set_ticklabels(ylabels)

ax_l.set_xlabel('gamma')
ax_l.set_ylabel('C')
ax_l.set_zlabel('Accuracy')
ax_l.set_title("SVM(linear)")
values = np.linspace(0.2, 1., xposM.ravel().shape[0])
colors = cm.rainbow(values)
ax_l.bar3d(xposM.ravel(), yposM.ravel(), dz*0, dx, dy, dz, color=colors)
plt.show()

#poly
fig = plt.figure(figsize=(5, 5), dpi=150)
ax_p = fig.add_subplot(111, projection='3d')
poly_result = np.array(poly_result)
zpos = poly_result
zpos = zpos.ravel()
dz = zpos

ax_p.w_xaxis.set_ticks(xpos + dx/2.)
ax_p.w_xaxis.set_ticklabels(xlabels)

ax_p.w_yaxis.set_ticks(ypos + dy/2.)
ax_p.w_yaxis.set_ticklabels(ylabels)

ax_p.set_xlabel('gamma')
ax_p.set_ylabel('C')
ax_p.set_zlabel('Accuracy')
ax_p.set_title("SVM(poly)")

ax_p.bar3d(xposM.ravel(), yposM.ravel(), dz*0, dx, dy, dz, color=colors)
plt.show()

#rbf
fig = plt.figure(figsize=(5, 5), dpi=150)
ax_r = fig.add_subplot(111, projection='3d')
rbf_result = np.array(rbf_result)
zpos = rbf_result
zpos = zpos.ravel()
dz = zpos

ax_r.w_xaxis.set_ticks(xpos + dx/2.)
ax_r.w_xaxis.set_ticklabels(xlabels)

ax_r.w_yaxis.set_ticks(ypos + dy/2.)
ax_r.w_yaxis.set_ticklabels(ylabels)

ax_r.set_xlabel('gamma')
ax_r.set_ylabel('C')
ax_r.set_zlabel('Accuracy')
ax_r.set_title("SVM(rbf)")

ax_r.bar3d(xposM.ravel(), yposM.ravel(), dz*0, dx, dy, dz, color=colors)
plt.show()

#sigmoid
fig = plt.figure(figsize=(5, 5), dpi=150)
ax_s = fig.add_subplot(111, projection='3d')
sigmoid_result = np.array(sigmoid_result)
zpos = sigmoid_result
zpos = zpos.ravel()
dz = zpos

ax_s.w_xaxis.set_ticks(xpos + dx/2.)
ax_s.w_xaxis.set_ticklabels(xlabels)

ax_s.w_yaxis.set_ticks(ypos + dy/2.)
ax_s.w_yaxis.set_ticklabels(ylabels)

ax_s.set_xlabel('gamma')
ax_s.set_ylabel('C')
ax_s.set_zlabel('Accuracy')
ax_s.set_title("SVM(sigmoid)")

ax_s.bar3d(xposM.ravel(), yposM.ravel(), dz*0, dx, dy, dz, color=colors)
plt.show()


#=======================================================================================
#Ensemble with VotingClassifier
ensemble_clf = VotingClassifier(estimators=[
    ('ran', ran_clf), ('svm', svm_clf), ('reg', reg_clf)], voting='hard')

ensemble_clf = ensemble_clf.fit(X_train_D1, y_train_D1)
y_pred = ensemble_clf.predict(X_test_D2)


print("\n\n\nEnsemble classifier with VotingClassifier")
ensemble_acc = accuracy_score(y_test_D2, y_pred)
print("Accuracy:", ensemble_acc)

#Get the confusion matrix
cf_matrix = confusion_matrix(y_test_D2, y_pred)
print("\nEnsemble with VotingClassifier Confusion Matrix\n", cf_matrix)
sns.heatmap(cf_matrix, annot=True, cmap="YlGnBu")
plt.title("Ensemble with VotingClassifier Confusion Matrix")
plt.show()


#=======================================================================================
#Ensemble without VotingClassifier

y_pred_ran = ran_clf.predict(X_test_D2)
y_pred_reg = reg_clf.predict(X_test_D2)
y_pred_svm = svm_clf.predict(X_test_D2)

y_pred_final = []
for i in range(len(X_test_D2)):
    temp = [y_pred_ran[i], y_pred_reg[i], y_pred_svm[i]]

    vote = Counter(temp)
    y_pred_final.append(vote.most_common(1)[0][0])

print("\n\n\nEnsemble classifier without VotingClassifier")
ensemble_without_acc = accuracy_score(y_test_D2, y_pred_final)
print("Accuracy:", ensemble_without_acc)

#Get the confusion matrix
cf_matrix = confusion_matrix(y_test_D2, y_pred_final)
print("\nEnsemble without VotingClassifier Confusion Matrix\n", cf_matrix)
sns.heatmap(cf_matrix, annot=True, cmap="YlGnBu")
plt.title("Ensemble without VotingClassifier Confusion Matrix")
plt.show()



#=======================================================================================
print("\n\n\nAccuracy comparison")
print("Logistic Regression accuracy: ", reg_acc)
print("Random Forest accuracy: ", ran_acc)
print("SVM accuracy: ", svm_acc)
print("Ensemble with Voting Classifier accuracy: ", ensemble_acc)
print("Ensemble without Voting Classifier accuracy: ", ensemble_without_acc)