# importing library
import pandas as pd
from featurewiz import featurewiz
import matplotlib.pyplot as plt
import yaml

feature_count = yaml.safe_load(open("params.yaml"))["no_of_features"]
# reading csv file
df = pd.read_csv("./data/Full_Features.csv")

# listing all columns in dataframe
all_column = df.columns

# extracting the usefull feature using featurewiz
features = featurewiz(df, target='Label', corr_limit=0.70, verbose=0)

# generating updated dataframe using important features
x= pd.DataFrame()
count = 1
for i in features[0]:
    if count <= feature_count:
        x[i]=df[i]
    else:
        break
    count +=1
x["Label"] = df.iloc[:,-1:]


# saving csv file
x.to_csv("./data/Feature_Importance.csv")