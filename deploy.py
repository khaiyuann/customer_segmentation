# -*- coding: utf-8 -*-
"""
Created on Wed May 18 15:18:02 2022

This script is used to deploy the trained model for customer segmentation.

@author: LeongKY
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer
from tensorflow.keras.models import load_model

from segmentation_modules import SegmentClassifier

#%% Static paths
TRAIN_PATH = os.path.join(os.getcwd(), 'datasets', 'train.csv')
DATA_PATH = os.path.join(os.getcwd(), 'datasets', 'new_customers.csv')
MODEL_LOAD_PATH = os.path.join(os.getcwd(), 'saved_model','model.h5')
ENC_PATH = os.path.join(os.getcwd(), 'saved_model', 'encoder.pkl')
segment_dict = {0:'A', 1:'B', 2:'C', 3:'D'}

#%% Load model and encoders
classifier = load_model(MODEL_LOAD_PATH)
classifier.summary()

encoder = pickle.load(open(ENC_PATH, 'rb'))

seg = SegmentClassifier()
df = pd.read_csv(DATA_PATH)
cust = df.copy()
train = pd.read_csv(TRAIN_PATH)

#%% Extract and encode features
cols = ['Ever_Married', 'Age', 'Graduated', 'Profession', 'Family_Size',
        'Spending_Score']
cols_cat = ['Ever_Married', 'Graduated', 'Profession', 'Spending_Score']
cols_num = ['Age', 'Family_Size']
cust = cust[cols]

#impute NaN using defined function
imputer_num = KNNImputer(n_neighbors=5)
imputer_cat = SimpleImputer(strategy='most_frequent')
seg.imputer(cust, cols_cat, imputer_cat)
seg.imputer(cust, cols_num, imputer_num)

#transform features using saved encoders
for col in cust[['Ever_Married', 'Graduated', 'Profession', 'Spending_Score']]:
    with open(os.path.join(os.getcwd(), 'saved_model','enc_'+str(col)+'.pkl'),
              'rb') as file:
        enc = pickle.load(file)
    cust[col] = enc.transform(cust[col])

#%% Predict segment using model
y_pred = np.argmax(classifier.predict(cust), axis=1)

#decode and append predictions
df['Segmentation'] = pd.DataFrame(list(map(segment_dict.get, y_pred)), 
                                  columns=['Segmentation'])

# export decoded predictions to new_customers csv file
df.to_csv(DATA_PATH)