'''
Created on Jul 25, 2020

@author: johan kleuskens
'''
import pandashelper as pdh
import pandas as pd
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras

def preprocess_dataset(df):
    # Drop passenger id
    df = df.drop('PassengerId', 1)
    # Change Pclass into dummy vars
    pdh.encode_text_dummy(df, 'Pclass')
    #Drop Name of passenger
    df = df.drop('Name', 1)
    # Change Sex into dummy vars
    pdh.encode_text_dummy(df, 'Sex')
    # Fill in missing values on Age column
    pdh.missing_median(df, 'Age')
    # Normalize Age
    pdh.encode_numeric_zscore(df, 'Age')
    # Normalize nr siblings
    pdh.encode_numeric_zscore(df, 'SibSp')
    # Normalize nr of parents
    pdh.encode_numeric_zscore(df, 'Parch')
    # Remove ticket
    df = df.drop('Ticket', 1)
    # Normalize Fare and remove outliers
    pdh.encode_numeric_zscore(df, 'Fare')
    pdh.remove_outliers(df, 'Fare', 3)
    # Remove cabin
    df = df.drop('Cabin', 1)
    # Change place of embarked into dummy vars
    pdh.encode_text_dummy(df, 'Embarked')
    return df
    
    
# // Load the training and test set 
dataset_training = pd.read_csv("~/Titanic-AI/datasets/train.csv")
dataset_test = pd.read_csv("~/Titanic-AI/datasets/test.csv")

# preprocess the dataset_training
dataset_training = preprocess_dataset(dataset_training)

# Convert train set to numpy arrays
dataset_training_x,  dataset_training_y = pdh.to_xy(dataset_training, 'Survived')
# We need to overwrite the y dataset here becuase the to_xy function transforms 'Survived' into a dummy var
dataset_training_y = dataset_training['Survived']  

# Split data set into train and test set
train_x, test_x, train_y, test_y = train_test_split(dataset_training_x, dataset_training_y, test_size = 0.15, random_state = 0)

print('Finished preprocessing data...')

# Create a Keras model
model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(12, activation=tf.nn.relu),
        tf.keras.layers.Dense(16,activation=tf.nn.relu),
        tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
    ])
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# Train the model
model.fit(train_x, train_y, epochs=50)

# Getting prediction data from test set
prediction = model.evaluate(test_x, test_y)
         