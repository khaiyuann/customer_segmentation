# -*- coding: utf-8 -*-
"""
Created on Wed May 18 14:19:08 2022

This class file contains the functions for performing a multiclass
customer segmentation of 4 categories for an automobile company.

@author: LeongKY
"""
import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import plot_model
from sklearn.metrics import confusion_matrix, classification_report,\
    accuracy_score


class SegmentClassifier():
    def __init_(self):
        pass
    
    def imputer(self, df, columns, method):
        '''
        This function is used to impute missing data.

        Parameters
        ----------
        df : DataFrame
            DataFrame containing columns to impute.
        columns : list
            List containing columns with missing data.
        method : Imputer object
            Selected imputer to fill missing values.

        Returns
        -------
        None.

        '''
        for col in columns:
            df[col] = pd.DataFrame(method.fit_transform
                                   (np.expand_dims
                                    (df[col], -1)))
    
    def cat_feature_encoding(self, df, columns, method):
        '''
        This function is used to perform label encoding on categorical
        features and save the resulting encoding in .pkl format for every
        encoded column.

        Parameters
        ----------
        df : DataFrame
            DataFrame containing features to encode.
        columns : list
            List containing categorical columns.
        method : Encoder object
            Selected encoder.

        Returns
        -------
        None.

        '''
        for col in df[columns]:
            df[col] = method.fit_transform(np.expand_dims(df[col], -1))
            pickle.dump(method, open(os.path.join(os.getcwd(), 
                                     'saved_model',
                                     'enc_'+str(col)+'.pkl'), 'wb'))
            
    def check_corr(self, X, y, label):
        '''
        This function is used to evaluate the correlation between features
        including categorical features after encoding.

        Parameters
        ----------
        X : DataFrame
            encoded DataFrame containing features.
        y : DataFrame/Array
            DataFrame/Array containing targets.
        label : str
            name of target column.

        Returns
        -------
        None.

        '''
        y_corr = pd.DataFrame(y, columns=[label])
        df_corr = X.join(y_corr)
        correlation = df_corr.corr()
        sns.heatmap(abs(correlation), annot=True, cmap=plt.cm.Reds)
        plt.show()
        
    def target_ohe(self, target, path):
        '''
        This function is used to encode the target using One Hot Encoding.

        Parameters
        ----------
        target : DataFrame/Array
            DataFrame/Array containing target.
        path : path
            Directory to save encoder.

        Returns
        -------
        target : Array
            Array containing One Hot encoded target.

        '''
        encoder_tr = OneHotEncoder(sparse=False)
        target = encoder_tr.fit_transform(np.expand_dims(target, -1))
        pickle.dump(encoder_tr, open(path, 'wb'))
        return target
    
    def create_model(self, nodes, dropout, nb_class, shape):
        '''
        This function is used to instantiate a model with 2 hidden layers with
        Dropout.

        Parameters
        ----------
        nodes : int
            number of nodes for each hidden layer.
        dropout : float
            dropout parameter.
        nb_class : int
            number of output classes.
        shape : shape
            shape of model input.

        Returns
        -------
        model : model
            instantiated deep learning model.

        '''
        model = Sequential()
        model.add(Dense(nodes, activation='relu', input_shape=(shape)))
        model.add(Dropout(dropout))
        model.add(Dense(nodes, activation='relu'))
        model.add(Dropout(dropout))
        model.add(Dense(nb_class, activation='softmax'))
        plot_model(model, os.path.join(os.getcwd(), 'results', 'model.png'))
        model.summary()
        return model
    
    def evaluate_model(self, model, X_test, y_test):
        '''
        This function is used to evaluate the developed model using 
        confusion matrix, f1 score, and accuracy.

        Parameters
        ----------
        model : model
            developed deep learning model.
        X_test : DataFrame/Array
            test feature data.
        y_test : DataFrame/Array
            test target data.

        Returns
        -------
        None.

        '''
        y_pred = model.predict(X_test)

        # model scoring
        y_pred_res = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        # model evaluation
        print('\nConfusion Matrix:\n')
        print(confusion_matrix(y_true, y_pred_res))
        print('\nClassification Report:\n')
        print(classification_report(y_true, y_pred_res))
        print('\nThis model has an accuracy of ' 
              + str('{:.2f}'.format(accuracy_score(y_true, y_pred_res)*100))
              +' percent')