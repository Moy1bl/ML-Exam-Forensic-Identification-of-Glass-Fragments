import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from DecisionTreeClassifier import Decision_Tree_Classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

"""TRAIN"""
df = pd.read_csv("df_train.csv")
df_target = df["type"]
df_feats = df.drop(columns=["type"])

x_train = df_feats
y_train = df_target

"""TEST"""
df_test = pd.read_csv("df_test.csv")
df_test_target = df_test["type"]
df_test_feats = df_test.drop(columns=["type"])

x_test = df_test_feats
y_test = df_test_target

my_data = np.loadtxt("df_test.csv", delimiter=",", skiprows=1, usecols=range(0,9))


clf = Decision_Tree_Classifier(max_depth=6, min_sample_split= 2 )
clf.fit(x_train, y_train)
y_pred = list(map(lambda x: int(x), clf.predict(my_data)))
clf.print_tree()

accuracy_score = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy_score)



DTree_un = DecisionTreeClassifier(max_depth = 6, min_samples_split = 2)
DTree_un.fit(x_train, y_train)
#Predict test data set.
y_pred_lib = DTree_un.predict(x_test)
print(classification_report(y_test, y_pred_lib))

