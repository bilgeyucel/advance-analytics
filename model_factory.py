# -*- coding: utf-8 -*-
"""
Created on Tue May 10 09:27:38 2022

@author: tijl_
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pickle
import datetime as dt


#Useful packages for building deep neural networks. 
import tensorflow as tf 
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D,Flatten,Dense,Dropout, Reshape,MaxPooling2D,Conv2D, LSTM
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense

#Additional library which we will use for preprocessing our image data before training our model and to provide some specific evaluation metrics.
import sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

#Import sister scripts
from Data_preprocessing import *
from Helper_functions  import *
from automated_feature_engineering import *


#--- DENSE NETWORK ---
def build_dense_nn(input_shape):
    # from keras import Sequential
    model = Sequential()
    model.add(layers.Dense(256, activation="relu", input_dim=input_shape))
    model.add(Dropout(0.3))
    model.add(Dense(128,activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(32,activation="relu"))
    model.add(Dense(1,activation="sigmoid"))
    return model


#--- LSTM ---
def build_lstm(input_shape):
    x_input = Input(input_shape)
    x = x_input
    x = Conv1D(filters = 128, kernel_size=1, activation = 'linear')(x)
    x = Dropout(0.3)(x)
    x = LSTM(60, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(units= 128, activation = 'relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(units= 64, activation = 'relu')(x)
    x = Dense(1, activation='sigmoid')(x)

    # Create model
    model = Model(inputs = x_input, outputs = x)
    
    return model



class logistic_regression:
    def __init__(self ):
        self.clf = None
        self.features = None
        self.test = None
        
    def fit(self, features, y_train, class_weight=None):
        self.features = features
        self.clf = LogisticRegression(class_weight=class_weight, random_state=1)
        
        # train the model
        self.clf.fit(self.features, y_train)
#         print(f"The best parameters are {self.clf.best_params_} with a score of {self.clf.best_score_} on validation data")
    
    def get_test_predict(self, text):
        self.test = text
        pred_val = self.predict(self.test)
        return pred_val
    
    def get_test_predict_proba(self, text):
        self.test = text
        pred_prob = self.clf.predict_proba(self.test)
        return pred_prob
        
    def get_metrics(self, y_test, pred_val):
        print("Report for test data \n\n", classification_report(y_test, pred_val))
        
    def predict(self, X):
        return self.clf.predict(X)

    def __call__(self, X):
        return self.predict(X)
