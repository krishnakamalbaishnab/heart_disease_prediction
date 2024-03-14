import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

heart_disease = pd.read_csv("./heart-disease.csv")


heart_disease.head()

#create x (all the feature column)
X = heart_disease.drop("target", axis=1)

#Create Y (the  target column)
y = heart_disease["target"]

# Split the data into training and test sets

from sklearn.model_selection import train_test_split
X_train, X_test , y_train, y_test = train_test_split(X,y)

#view the data shapes

X_train.shape , X_test.shape , y_train.shape, y_test.shape

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()

#make prediction

y_predicts = model.predict(X_test)

y_predicts




heart_disease.loc[206]
# Make a prediction on a single sample (has to be array)
model.predict(np.array(X_test.loc[206]).reshape(1, -1))

# On the training set
model.score(X_train, y_train)
# On the test set (unseen)
model.score(X_test, y_test)

# Try different numbers of estimators (n_estimators is a hyperparameter you can change)
np.random.seed(42)
for i in range(10, 100, 10):
    print(f"Trying model with {i} estimators...")
    model = RandomForestClassifier(n_estimators=i).fit(X_train, y_train)
    print(f"Model accruacy on test set: {model.score(X_test, y_test)}")
    print("")


    #saving model for later use

import pickle
pickle.dump(model, open("random_forest_model.pkl","wb"))

#load the save model

loaded_model = pickle.load(open("random_forest_model.pkl","rb"))
loaded_model.predict(np.array(X_test.loc[206]).reshape(1,-1))