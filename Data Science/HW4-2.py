import pandas as pd
import numpy as np
import math
from sklearn import preprocessing

df = pd.read_excel('C:/python_file/k_nearest_dataset.xlsx',index_col=None)

#If the value of the T-shirt is M, 0 and L, change to 1
for i in range(len(df)):
    if df.loc[i,"T_shirt_size"] =='M':
        df.loc[i,"T_shirt_size"]=0
    elif df.loc[i,"T_shirt_size"] =='L':
        df.loc[i,"T_shirt_size"]=1

df_list=df.values.tolist()

#Enter your predicted height and weight
height=int(input("Enter the Height(cm): "))
weight=int(input("Enter the Weight(kg): "))
value_to_predict=[]
value_to_predict.append(height)
value_to_predict.append(weight)


scaler = preprocessing.StandardScaler()

#Normalize by inserting data to predict
df_temp=df.copy()
df_temp.loc[len(df)]=[height, weight,-1]
height = np.array(df_temp['Height(cm)'])
weight = np.array(df_temp['Weight(kg)'])
height = scaler.fit_transform(height.reshape(-1, 1))
df_temp['Height(cm)'] = height
weight = scaler.fit_transform(weight.reshape(-1, 1))
df_temp['Weight(kg)'] = weight
#Change Normalized Data to List
scaled_df_temp = df_temp.values.tolist()



#Normalize without Predicting Data
df_temp2=df.copy()
height = np.array(df_temp2['Height(cm)'])
weight = np.array(df_temp2['Weight(kg)'])
height = scaler.fit_transform(height.reshape(-1, 1))
df_temp2['Height(cm)'] = height
weight = scaler.fit_transform(weight.reshape(-1, 1))
df_temp2['Weight(kg)'] = weight
#Change Normalized Data to List
scaled_df = df_temp2.values.tolist()
scaled_dataframe= pd.DataFrame(df_temp2, columns=['Height(cm)','Weight(kg)',"T_shirt_size"])



#To find the distance between two points
def distance(p1,p2):
    d=math.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)
    return d

#The function of finding the k points closest to the target.
def find_neighbor(dataset, target, k):
    distance_list=list()
    for point in dataset:
        d=distance(target, point)
        distance_list.append((point,d))

    #Sort using distance.
    distance_list.sort(key=lambda tup: tup[1])
    neighbor_list=list()

    for i in range(k):
        neighbor_list.append(distance_list[i][0])
    return neighbor_list




#A function that predicts using the values of the neighbors found,
def predict(dataset, target, k):
    #Find nearby k neighborhood data
    neighbor_list=find_neighbor(dataset, target, k)
    print("\nNearest",k,"data")
    for i in neighbor_list:
        print(i)
    output=[]
    for i in neighbor_list:
        output.append(i[-1])
    prediction = max(set(output), key=output.count)
    #Change the pre-diction value to size
    if prediction==0:
        size="M"
    elif prediction==1:
        size="L"
    return size



prediction = predict(scaled_df, scaled_df_temp[-1], 3)
print('\nIt was predicted that a person with \na height {}cm and a weight {}kg would have to wear a T-shirt size {}.'.format(value_to_predict[0],value_to_predict[1],prediction))
