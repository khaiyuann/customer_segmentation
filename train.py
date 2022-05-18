# -*- coding: utf-8 -*-
"""
Created on Wed May 18 09:19:20 2022

This script is used to train a deep learning model for customer segmentation.

@author: LeongKY
"""

#%% Imports
import os
from datetime import datetime 
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from segmentation_modules import SegmentClassifier

#%% Static paths
DATA_PATH = os.path.join(os.getcwd(), 'datasets', 'train.csv')
MODEL_SAVE_PATH = os.path.join(os.getcwd(), 'saved_model','model.h5')
SCALER_PATH = os.path.join(os.getcwd(), 'saved_model', 'scaler.pkl')
ENC_PATH = os.path.join(os.getcwd(), 'saved_model', 'encoder.pkl')
LOG_PATH = os.path.join(os.getcwd(), 'logs')

#%% 1. Load data
df = pd.read_csv(DATA_PATH)

#%% 2. Inspect data
print(df.describe().T)
print(df.info())
print(df.duplicated().sum())

#found NaN values in columns 2, 4, 5, 6, 8 and 9 and no duplicated data
#%% 3. Clean data
# convert categorical features to 'category' type
cat_col = ['Gender', 'Ever_Married', 'Graduated', 'Profession',
           'Spending_Score', 'Var_1']
df[cat_col] = df[cat_col].astype('category')

#drop 'ID' column before imputation as not meaningful for analysis
df = df.drop(labels='ID', axis=1)

# instantiate class containing functions
seg = SegmentClassifier()

#use imputation to fill NaN as more than 5% of data loss if drop NaN rows
#use KNN imputer with 5 neighbors to impute found numerical NaN values
imputer_num = KNNImputer(n_neighbors=5)
nan_num = ['Work_Experience', 'Family_Size']
seg.imputer(df, nan_num, imputer_num)
    
#use SimpleImputer to impute found categorical NaN values
imputer_cat = SimpleImputer(strategy='most_frequent')
nan_cat = ['Ever_Married', 'Graduated', 'Profession', 'Var_1']
seg.imputer(df, nan_cat, imputer_cat)

#%% 4. Data preprocessing & feature selection
#in same section as encoding is required for feature selection

# separate features and labels
X = df.drop(labels=['Segmentation'], axis=1)
y = df['Segmentation']

# encoding categorical features
encoder = LabelEncoder()
seg.cat_feature_encoding(X, cat_col, encoder)

# encoding labels for feature selection
y_sel = encoder.fit_transform(np.expand_dims(y, -1))

# correlation heatmap for feature selection
seg.check_corr(X, y_sel, 'Segmentation')

# based on correlation, select features with correlation >= 0.1
X = X[['Ever_Married', 'Age', 'Graduated', 'Profession', 'Family_Size', 
        'Spending_Score']]

# one hot encoding labels for model training
y = seg.target_ohe(y, ENC_PATH)

# split train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=23)

#%% 5. Create model
model = seg.create_model(nodes=512, dropout=0.3, 
                         nb_class=4, shape=X_train.shape[1:])

# callbacks
log_files = os.path.join(LOG_PATH, datetime.now().strftime('%Y%m%d-%H%M%S'))

es_callback = EarlyStopping(monitor='loss', patience=5)
tb_callback = TensorBoard(log_dir=log_files)
callbacks = [es_callback, tb_callback]

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='acc')

# train model
hist = model.fit(X_train, y_train, epochs=100,
                 validation_data=(X_test, y_test),
                 callbacks=callbacks)

#%% 7. Evaluate model
seg.evaluate_model(model, X_test, y_test)

# export model for deployment
model.save(MODEL_SAVE_PATH)

#%% Analysis of results
'''
From the results obtained of 52.29%, it falls short of the targeted 80%.
There are various limitations that contribute to the result deficiencies:
    
    1. The dataset consists of many categorical features that have NaN
        data, and to perform precise imputation on the dataset will be costly
        in both time and computational power, hence only a simple imputer
        was used to complete the model within the available time and resources.
        
    2. Customer segmentation is a multiclass classification problem that
        is highly complex and difficult to achieve high accuracy on, especially
        given point (1) where the data provided consists of many categories
        each with multiple items and missing data. Simply dropping the data
        is not an option due to the varied nature of the data, however
        imputation as a solution is not optimal as it is unable to replicate
        actual data. Hence, the lower accuracy obtained can be attributed to
        the nature of the dataset and is limited by such.
        
    3. The correlation indices obtained from the correlation heatmap have very
        low correlation, with the higest of 0.24 for the 'Age' column. The
        overall low correlation between features and the target heavily limits
        the potential of the DL model to accurately predict the segment of a
        customer. If visualized, it would show clusters of the customer 
        segment in very close proximity and without clear borders, as the
        features used in training does not strongly correlate to a segment.
        Hence, this will result in a high quantity of wrong predictions.
        
Future improvements to the accuracy of the model can be achieved by
employing more sophisticated techniques and DL architectures, or machine
learning approaches may be more effective for this particular problem. These
proposed solutions are to be considered on future work concerning this 
customer segmentation dataset.
'''