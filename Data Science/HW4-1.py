import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from pprint import pprint
import math

#Read the file
df = pd.read_excel('C:/python_file/decision_tree_dataset.xlsx','dataset', index_col=None)

#pridictorattribute
predictor_attribute = ["District","House Type","Previous Customer"]
#target attribute
target_attribute= ["Outcome"]



#entropy calculation function
def entropy(target):
    elements, count = np.unique(target, return_counts = True)
    entropy=0 
    for i in range(len(elements)):
        p=count[i]/np.sum(count)
        entropy+=p* math.log(p,2)
    entropy*=(-1)
    return entropy



#Function to receive information gain
def Gain(data,split_attribute, target):
    # Calculate root entropy
    entropy_root = entropy(data[target])
    
    #child weighted entropy calculation
    node,count = np.unique(data[split_attribute],return_counts=True)
    entropy_child=0
    for i in range(len(node)):
        p=count[i]/np.sum(count)
        entropy_child += p*entropy(data.where(data[split_attribute]==node[i]).dropna()[target])

    print('Entropy(', split_attribute, ') = ', round(entropy_child, 3))

    # Calculate Information Gain
    Information_Gain = entropy_root - entropy_child
    return Information_Gain
 
 

#Function to make tree
def makeTree(data, original_data, features, target, parent_node_class = None):

    #Returns a property if you have a single value
    if len(np.unique(data[target])) <= 1:
        return np.unique(data[target])[0]

    #Returns the property with the maximum value in the original data when data is missing
    elif len(data)==0:
        return np.unique(original_data[target])[np.argmax(np.unique(original_data[target], return_counts=True)[1])]
 
    elif len(features) ==0:
        return parent_node_class

    #make tree
    else:
        #parent
        parent_node_class = np.unique(data[target])[np.argmax(np.unique(data[target], return_counts=True)[1])]

        #Select Split Properties
        item=[]
        for i in features:
            print("Split attribute: ", i)
            item.append(Gain(data,i,target))
            print("InfoGain(",i," ) = ", round(Gain(df,i, "Outcome"), 3), '\n')
        index_best = np.argmax(item)
        bestF = features[index_best]

        #tree structure
        tree = {bestF:{}}
        
        features = [i for i in features if i != bestF]

        #The branches grow.
        for value in np.unique(data[bestF]):
            subData = data.where(data[bestF] == value).dropna()
            subtree = makeTree(subData, data, features, target, parent_node_class)
            tree[bestF][value] = subtree
            
        return(tree)

tree = makeTree(df, df, predictor_attribute, target_attribute)
print("\Result of Decision Tree")
pprint(tree)


