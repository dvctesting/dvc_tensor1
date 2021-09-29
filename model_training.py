import numpy as np
import pandas as pd
import json
from sklearn import metrics
from sklearn import tree
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from dtreeviz.trees import *


import os
os.environ["PATH"] += os.pathsep + '/usr/lib/x86_64-linux-gnu/graphviz'

# reading csv file
data = pd.read_csv("./data/Feature_Importance.csv")

# Create independent and Dependent Features
# listing all column in the dataframe
columns = data.columns.tolist()

# Filter the columns to remove data we do not want 
columns = [c for c in columns if c not in ["Label"]]

# Store the variable we are predicting 
target = "Label"
X = data[columns]
Y = data[target]



# Balanced the inbalanced dataset
oversample =  RandomOverSampler(sampling_strategy='minority')
X_res, y_res = oversample.fit_resample(X, Y)


# train test split
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3)

# Using Random forest classifier to create classification model 
model = tree.DecisionTreeClassifier()
model.fit(X_train,y_train)

# Pridicting the Y values using X_test
y_pred=model.predict(X_test)
# Comparing and finding the accurecy of model
acc= metrics.accuracy_score(y_test, y_pred)
print(acc)
# viz = dtreeviz(model,
#                 X_train,
#                 y_train,
#                 feature_names = columns, class_names= [ 7,  6,  4,  1, 10,  8, 13])

# viz.save("dtree.svg")


# fig, axes = plt.subplots(figsize = (3,3), dpi=100)
figure = plt.gcf()

tree.plot_tree(model,max_depth=3)
figure.savefig('dgraph.png',dpi=100)

# Saving accuracy to the metrics.json file
with open("metrics.json", 'w') as outfile:
    json.dump({ "accuracy":acc }, outfile)

