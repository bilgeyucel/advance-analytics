# -*- coding: utf-8 -*-
"""
Created on Tue May 10 09:12:14 2022

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
from tensorflow.keras.layers import Conv1D,Flatten,Dense,Dropout, Reshape,MaxPooling2D,Conv2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense

#Additional library which we will use for preprocessing our image data before training our model and to provide some specific evaluation metrics.
import sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.preprocessing import RobustScaler
import tensorflow.keras.backend as K

#Import sister scripts
from Data_preprocessing import *
from model_factory import *
from automated_feature_engineering import *


#--- Helper Functions ---
def get_age_in_months(d1, d2):
    d1 = dt.datetime.strptime(d1, "%Y-%m")
    d2 = dt.datetime.strptime(d2, "%Y-%m")
    return int(abs((d2 - d1).days/30))

def printMissingData(df):
  features = df.columns.values.tolist()

  print("Number of rows:",len(df))
  print("Missing data")
  print("------------")
  for feature in features:
      missing = (df[feature].isnull()).sum()
      if missing != 0:
          print(feature,':',missing,'->',(missing/len(df))*100,'%')

def getRegionClass(postalCodes):
    Brussels = []
    Flanders = []
    Wallonia = []
    Other = []
    for code in postalCodes:
        #Brussels
        if ((code >= 1000) and (code <= 1212)) or ((code >= 1931) and (code <= 1950)):
            Brussels.append(1)
            Flanders.append(0)
            Wallonia.append(0)
            Other.append(0)
        #Flanders
        elif ((code >= 1500) and (code <= 4690)) or ((code >= 8000) and (code <= 9999)):
            Brussels.append(0)
            Flanders.append(1)
            Wallonia.append(0)
            Other.append(0)
        #Wallonia
        elif (code >= 4000) and (code <= 7970):
            Brussels.append(0)
            Flanders.append(0)
            Wallonia.append(1)
            Other.append(0)
        #Other
        else:
            Brussels.append(0)
            Flanders.append(0)
            Wallonia.append(0)
            Other.append(1)
    return Brussels,Flanders,Wallonia,Other

def normalize_minmax(column):
    a,b = 0,1
    min,max = column.min(),column.max()
    return (column-min) / (max - min)
  
def normalize_robust(column):
    trans = RobustScaler()
    print('\nmax before transform: {}'.format(max(column)))
    column = pd.DataFrame(column) #make 2D for robustscaler input
    column = trans.fit_transform(column)
    print(column.shape)
    column = pd.DataFrame(column).squeeze() #make 2D again
    print('\nmax after transform: {}'.format(max(column)))
    return column



#CUSTOM LOSS FUNCTIONS
ALPHA = 0.8
BETA = 0.2

def TverskyLoss(targets, inputs, alpha=ALPHA, beta=BETA, smooth=1e-6):
        
    #flatten label and prediction tensors
    inputs = tf.cast(K.flatten(inputs), dtype="float32")
    targets =tf.cast(K.flatten(targets), dtype="float32")
    
    #True Positives, False Positives & False Negatives
    TP = K.sum((inputs * targets))
    FP = K.sum(((1-targets) * inputs))
    FN = K.sum((targets * (1-inputs)))
   
    Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
    
    return 1 - Tversky

def DiceLoss(targets, inputs, smooth=1e-6):
    
    #flatten label and prediction tensors
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)
    
    intersection = K.sum(K.dot(targets, inputs))
    dice = (2*intersection + smooth) / (K.sum(targets) + K.sum(inputs) + smooth)
    return 1 - dice
  

#Create file to submit to leaderboard
def generate_submission(client_ids,X_test,model,filename):
    
    predictions_proba = np.squeeze(model.predict(X_test))
    
    #Generate data frame to submit
    submission_arr = np.vstack((client_ids,predictions_proba)).T
    submission_df = pd.DataFrame(submission_arr)
    submission_df = submission_df.rename(columns={0: 'ID', 1: 'PROB'})
    print(submission_df.shape, submission_df.head)
    
    #Write submission dataframe to csv
    from datetime import date
    import time
    today = date.today()
    today = today.strftime('%y%m%d') 
    time_now = int(time.time())
    suffix = str(today) + '_' + str(time_now)
    filepath = "submissions/{}_{}".format(filename,suffix)
    print("Path for submission file : {}".format(filepath))
    
    submission_df.to_csv(filepath,sep=',',header=True,index=False)
