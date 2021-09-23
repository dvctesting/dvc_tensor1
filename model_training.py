# importing library
import numpy as np
import pandas as pd
import json
from sklearn import metrics
from imblearn.over_sampling import RandomOverSampler
from imblearn.combine import SMOTETomek
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import yaml



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
clf=RandomForestClassifier(n_estimators=100)
clf.fit(X_train,y_train)

# Pridicting the Y values using X_test
y_pred=clf.predict(X_test)
# Comparing and finding the accurecy of model
acc= metrics.accuracy_score(y_test, y_pred)

# Saving accuracy to the metrics.json file
with open("metrics.json", 'w') as outfile:
    json.dump({ "accuracy":acc }, outfile)

