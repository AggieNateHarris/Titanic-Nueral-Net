import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
import tflearn
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.estimator import regression


# Load the dataset and model
###############################################
# path for training & test sets
PATH = 'C:\\Users\\Nate\\Documents\\School\\cs5600\\Final Project\\titanic\\'
# load training set
training_set = pd.read_csv(PATH + "train.csv")
print("Training set shape: ")
print(training_set.shape)
print(training_set.head())

# load test set
testing_set = pd.read_csv(PATH + "test.csv")
print("Testing set shape: ")
print(testing_set.shape)
print(testing_set.head())


# Load saved model
with open('best_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)
###############################################

# Training and saving the network
###############################################
# Split some of the training data for validation testing
train, valid = train_test_split(training_set, test_size=0.2, random_state=42, shuffle=False)
print("Training subset after train-valid split: ")
print(train.shape)
print("Valid subset after train-valid split: ")
print(valid.shape)

# Model creation
working_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

# Select features from the pool of features for x and set label y
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]
train_X = train[features]
valid_X = valid[features]
test_X = testing_set[features]
train_Y = train["Survived"]
valid_Y = valid["Survived"]
# Use pandas get dummies for 1 hot encoding for non numerical values
train_X = pd.get_dummies(train_X)
train_X = train_X.fillna(0)
valid_X = pd.get_dummies(valid_X)
valid_X = valid_X.fillna(0)
test_X = pd.get_dummies(test_X)
test_X = test_X.fillna(0)
print(train_X.head())
print(test_X.head())

print("Training subset after train-valid split & feature selection: ")
print(train_X.shape)
print(train_Y.shape)
print("Valid subset after train-valid split & feature selection: ")
print(valid_X.shape)
print(valid_Y.shape)

# Model training and validation
# working_model.fit(train_X, train_Y)

# set working model to the loaded model if not training
working_model = loaded_model

predictions = working_model.score(valid_X, valid_Y)
print("Finished training and testing: ")
print(predictions)


# Save the model
val = input("Do you want to save the model? Y/N")
if val == "Y":
    with open('best_model.pkl', 'wb') as f:
        pickle.dump(working_model, f)
###############################################

# Use the prediction model on the test dataset and see the results
###############################################
# Use loaded model on the loaded test set
test_predictions = working_model.predict(test_X)

# Create csv file that can be submitted on kaggle and scored for accuracy
val2 = input("Do you want to create a submission file? Y/N")
if val2 == "Y":
    output = pd.DataFrame({'PassengerId': testing_set.PassengerId, 'Survived': test_predictions })
    output.to_csv(PATH + 'submission.csv', index=False)
###############################################
