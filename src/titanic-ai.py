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
    if 'Survived' in df:
        # Change type of Survived column to bool to prevent to_xy function to change this column into dummy vars
        df = df.astype({'Survived': bool})
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
    if 'Survived' in df:
        pdh.remove_outliers(df, 'Fare', 3)
    # Remove cabin
    df = df.drop('Cabin', 1)
    # Change place of embarked into dummy vars
    pdh.encode_text_dummy(df, 'Embarked')
    return df
    
    
# // Load the training and test set 
dataset_training_org = pd.read_csv("~/Titanic-AI/datasets/train.csv")
dataset_test_org = pd.read_csv("~/Titanic-AI/datasets/test.csv")

# preprocess the datasets
dataset_training = preprocess_dataset(dataset_training_org)
dataset_test = preprocess_dataset(dataset_test_org)

# Convert data sets to numpy arrays
dataset_training_x,  dataset_training_y = pdh.to_xy(dataset_training, 'Survived')
dataset_test_x = pdh.to_xy(dataset_test)

# Split training data set into train and test set
train_x, test_x, train_y, test_y = train_test_split(dataset_training_x, dataset_training_y, test_size = 0.15, random_state = 0)

print('Finished preprocessing data...')

# Create a Keras model
model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(input_shape = (12,), units = 16, activation=tf.nn.relu),
        tf.keras.layers.Dense(16,activation=tf.nn.relu),
        tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
    ])
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Train the model
#model.fit(train_x, train_y, epochs=50)
# Getting evaluation data from test set
#prediction = model.evaluate(test_x, test_y)

# Now use the complete dataset_training for training
model.fit(dataset_training_x, dataset_training_y, epochs=50)

# Do prediction for test set 
prediction = model.predict(dataset_test_x)

# Create a new dataframe
submission = pd.DataFrame(dataset_test_org['PassengerId'].copy())
submission['Survived'] = [0 if i < 0.5 else 1 for i in prediction]
submission.to_csv("~/Titanic-AI/datasets/submission.csv", index = False)

print('Finished predicting test set...')


         