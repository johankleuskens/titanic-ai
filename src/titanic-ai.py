'''
Created on Jul 25, 2020

@author: johan kleuskens
'''
import pandashelper as pdh
import pandas as pd
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras

# // Load the training and test set 
dataset_training = pd.read_csv("~/Titanic-AI/datasets/train.csv")
dataset_test = pd.read_csv("~/Titanic-AI/datasets/test.csv")

# Drop passenger id
dataset_training = dataset_training.drop('PassengerId', 1)
# Change Pclass into dummy vars
pdh.encode_text_dummy(dataset_training, 'Pclass')
#Drop Name of passenger
dataset_training = dataset_training.drop('Name', 1)
# Change Sex into dummy vars
pdh.encode_text_dummy(dataset_training, 'Sex')
# Fill in missing values on Age column
pdh.missing_median(dataset_training, 'Age')
# Normalize Age
pdh.encode_numeric_zscore(dataset_training, 'Age')
# Normalize nr siblings
pdh.encode_numeric_zscore(dataset_training, 'SibSp')
# Normalize nr of parents
pdh.encode_numeric_zscore(dataset_training, 'Parch')
# Remove ticket
dataset_training = dataset_training.drop('Ticket', 1)
# Normalize Fare and remove outliers
pdh.encode_numeric_zscore(dataset_training, 'Fare')
pdh.remove_outliers(dataset_training, 'Fare', 3)
# Remove cabin
dataset_training = dataset_training.drop('Cabin', 1)
# Change place of embarked into dummy vars
pdh.encode_text_dummy(dataset_training, 'Embarked')

# Convert train set to numpy arrays
dataset_training_x,  dataset_training_y = pdh.to_xy(dataset_training, 'Survived')
        
# Split data set into train and test set
train_x, test_x, train_y, test_y = train_test_split(dataset_training_x, dataset_training_y, test_size = 0.15, random_state = 0)

print('Finished preprocessing data...')

# Create a Keras model
model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(12, activation=tf.nn.relu),
        tf.keras.layers.Dense(32,activation=tf.nn.relu),
        tf.keras.layers.Dense(2, activation=tf.nn.sigmoid)
    ])
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# Train the model
model.fit(train_x, train_y, epochs=50)

# Getting prediction data from test set
prediction = model.evaluate(test_x, test_y)
         