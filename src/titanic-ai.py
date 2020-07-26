'''
Created on Jul 25, 2020

@author: johan
'''
import pandashelper as pdh
import pandas as pd
#from tensorflow_estimator.python.estimator.training import TrainSpec

# // Load the training and test set 
dataset_training = pd.read_csv("~/Titanic-AI/datasets/train.csv")
dataset_test = pd.read_csv("~/Titanic-AI/datasets/test.csv")

# Change Pclass into dummy vars
dataset_training = pdh.encode_text_dummy(dataset_training, 'Pclass')
# Change Sex into dummy vars
dataset_training = pdh.encode_text_dummy(dataset_training, 'Sex')
# Fill in missing values
dataset_training = pdh.missing_median(dataset_training, 'Age')
# Normalize Age
dataset_training = pdh.encode_numeric_zscore(dataset_training, 'Age')
# Change nr of siblings on board  into dummy vars
dataset_training = pdh.encode_text_dummy(dataset_training, 'SibSp')
# Change nr of parents on board  into dummy vars
dataset_training = pdh.encode_text_dummy(dataset_training, 'Parch')
# Normalize Fare
dataset_training = pdh.encode_numeric_zscore(dataset_training, 'Fare')
# Change place of embarked into dummy vars
dataset_training = pdh.encode_text_dummy(dataset_training, 'Embarked')


         
# Split train set into train and test set
test_size = int(dataset_training.count()[0] * 0.15)  # testing set will be 15% of training set 
train_set = dataset_training.iloc[:-test_size,:].copy()
test_set  = dataset_training.iloc[-test_size:,:].copy()

# Convert pandas dataframe to numpy array
df_train_set = train_set.to_numpy()
df_test_set  = test_set.to_numpy()

         