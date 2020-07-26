'''
Created on Jul 25, 2020

@author: johan kleuskens
'''
import pandashelper as pdh
import pandas as pd
from sklearn.model_selection import import train_test_split
#from tensorflow_estimator.python.estimator.training import TrainSpec

# // Load the training and test set 
dataset_training = pd.read_csv("~/Titanic-AI/datasets/train.csv")
dataset_test = pd.read_csv("~/Titanic-AI/datasets/test.csv")

# Drop passenger id
dataset_training.drop('PassengerId')
# Change Pclass into dummy vars
pdh.encode_text_dummy(dataset_training, 'Pclass')
#Drop Name of passenger
dataset_training.drop('Name')
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
# Normalize Fare
pdh.encode_numeric_zscore(dataset_training, 'Fare')
# Change place of embarked into dummy vars
pdh.encode_text_dummy(dataset_training, 'Embarked')
        
# Split train set into train and test set
train_set, test_set = train_test_split(dataset_training, 0.15)

# Convert pandas dataframe to numpy array

         