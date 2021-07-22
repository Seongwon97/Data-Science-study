
flu = ['Y', 'N', 'Y', 'Y', 'Y', 'N', 'Y', 'Y', 'Y', 'Y', 'N', 'Y', 'Y', 'Y', 'N', 'Y', 'Y', 'Y', 'Y', 'Y', 'N', 'Y', 'Y', 'N', 'Y', 'N', 'Y', 'Y',  'Y', 'N']
fever = ['L', 'M', 'H', 'M', 'L', 'M', 'H',  'M', 'M', 'L', 'M', 'L', 'H', 'M', 'M', 'H', 'M', 'H', 'M','L', 'M', 'L', 'M', 'H', 'M','L', 'H', 'M','L', 'M']
sinus = ['Y', 'N', 'Y', 'Y', 'Y', 'N', 'Y', 'Y', 'Y', 'N', 'Y', 'Y', 'Y', 'N', 'N', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'N', 'Y', 'Y', 'N', 'Y', 'Y','Y', 'N']
ache = ['Y', 'N', 'N', 'N', 'Y', 'N', 'N', 'N','Y', 'N', 'N', 'N','Y', 'N', 'N', 'N','Y', 'N', 'N', 'N','Y', 'N', 'N', 'N','Y', 'N', 'N', 'N','Y', 'N']
swell = ['Y', 'N', 'Y', 'N', 'Y', 'N', 'Y', 'N', 'Y', 'N', 'Y', 'Y', 'N', 'Y', 'N', 'N', 'N', 'Y', 'N','Y', 'N', 'Y', 'N', 'Y', 'Y', 'N', 'Y', 'N','Y', 'N']
headache = ['N', 'N', 'Y', 'Y', 'N', 'Y', 'Y', 'N', 'N', 'N', 'Y', 'Y', 'N', 'Y', 'N', 'N', 'Y', 'N', 'Y', 'Y', 'N', 'N', 'Y', 'Y', 'N', 'N', 'Y', 'Y', 'N', 'N']

# Import LabelEncoder
from sklearn import preprocessing

#create labelEncoder
le = preprocessing.LabelEncoder()

# Convert string labels into numbers.
label = le.fit_transform(flu)
fever_encoded = le.fit_transform(fever)
sinus_encoded = le.fit_transform(sinus)
ache_encoded = le.fit_transform(ache)
swell_encoded = le.fit_transform(swell)
headache_encoded = le.fit_transform(headache)

features = list(zip(fever_encoded, sinus_encoded, ache_encoded, swell_encoded, headache_encoded))

#Import Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB
#Create a Gaussian Classifier
model = GaussianNB()

# Train the model using the training sets
model.fit(features, label)
#Predict Output
predicted = model.predict([[2, 1, 0, 0, 0]])
# fever:M, sinus:Y, ache:N, swell:N, headache:N

print('Predicted Value:', predicted)

if(predicted == 0):
    print('\nResult of predict : Not flu')
else:
    print('\nResult of predict : Flu')
