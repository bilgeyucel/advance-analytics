import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pickle
import datetime as dt

#Useful packages for building deep neural networks. 
import tensorflow as tf 
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv1D,Flatten,Dense,Dropout, Reshape,MaxPooling2D,Conv2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Input
import tensorflow.keras.backend as K

#Additional library which we will use for preprocessing our image data before training our model and to provide some specific evaluation metrics.
import sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.preprocessing import RobustScaler

#Import sister scripts
from Data_preprocessing import *
from Helper_functions  import *
from Feature_Selector import *
from model_factory import *
from automated_feature_engineering import *



#--------------- GET THE NECESSARY DATA FOR MODEL TRAINING -----------------
_, _, train_df = get_clean_train_data(transform="normal",nan_handler="drop")
_, _, test_df = get_clean_test_data(transform="normal",nan_handler="to_zero")
train_df.drop("client_id",axis=1,inplace=True)
test_ids = test_df["client_id"]


#--- Format Training Data ---
X_train, y_train = np.array(train_df.loc[:, train_df.columns != 'target']),np.array(train_df.loc[:,'target'])
X_test = test_df.drop("client_id",inplace=False,axis=1)
print('training features shape:{}'.format(X_train.shape))
print('training target shape:{}\n'.format(y_train.shape))



#--- Add Kmeans featurization to test and train set ---
kmf_train = KMeansFeaturizer(k = 4, target_scale = 10).fit(X_train, y_train)
kmf_test = KMeansFeaturizer(k = 4, target_scale = 10).fit(X_test, None)
cluster_feature_train = kmf_train.transform(X_train)
cluster_feature_test = kmf_test.transform(X_test)
X_train = np.concatenate((X_train,cluster_feature_train),axis=1)
X_test = np.concatenate((X_test,cluster_feature_test),axis=1)



# --- Apply over -and undersampling to handle mitigate class imbalance ---
# from imblearn.over_sampling import SMOTE
# from imblearn.under_sampling import RandomUnderSampler
# oversample = SMOTE(sampling_strategy=0.1) #Generate new examples
# undersample = RandomUnderSampler(sampling_strategy=0.15) #undersample until we get a more equal distribution
# X_train,y_train = oversample.fit_resample(X_train, y_train)
# X_train,y_train = undersample.fit_resample(X_train, y_train)
# print(X_train.shape,y_train.shape)


# --- Set class weights to further mitigate imbalance ---
classes = np.unique(y_train,return_counts=True)[0]
class_weights_arr = sklearn.utils.class_weight.compute_class_weight(class_weight = 'balanced', classes = classes, y = y_train)

class_weights_dict = {} #input to model.fit requires dictionary
for i in classes:
    class_weights_dict[i] = class_weights_arr[i]
print("target attribute weights to handle class imbalance:{}".format(class_weights_dict))


# --- Split of validation set ---
val_indices = np.random.choice(a = len(X_train), size = int(len(X_train)*0.1))
X_val, y_val = X_train[val_indices], y_train[val_indices]
X_train, y_train = np.delete(arr=X_train,obj=val_indices,axis=0), np.delete(arr=y_train,obj=val_indices,axis=0)



#--------------- BUILD MODEL -----------------
#build and fit model
model = logistic_regression()
model.fit(X_train, y_train, class_weight=class_weights_dict)

#Evaluate performance
model.get_metrics(y_val, model.get_test_predict(X_val))

#Generate submission
generate_submission(client_ids=test_ids,X_test=X_test,model=model,filename="log_reg_normal_manual")






