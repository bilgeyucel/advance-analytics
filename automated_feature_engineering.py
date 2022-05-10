# -*- coding: utf-8 -*-
"""
Created on Tue May 10 09:21:15 2022

@author: tijl_
"""

from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
from sklearn.cluster import KMeans

#Import sister scripts
from Data_preprocessing import *
from Helper_functions  import *
from Feature_Selector import *
from model_factory import *

# ----------------------------- K-Means Featurization (Feature engineering for machine learning - Zheng & Casari) ----------

class KMeansFeaturizer:
  def __init__(self, k = 100, target_scale = 5.0, random_state = None):
    self.k = k
    self.target_scale = target_scale
    self.random_state = random_state

  def fit(self, X, y = None):
    if y is None:
      km_model = KMeans(n_clusters=self.k,
                        n_init = 20,
                        random_state = self.random_state)
      km_model.fit(X)
      
      self.km_model = km_model
      self.cluster_centers_ = km_model.cluster_centers_
      return self

    data_with_target = np.hstack((X, y[:,np.newaxis]*self.target_scale))
    km_model_pretrain = KMeans(n_clusters=self.k,
                               n_init=20,
                               random_state=self.random_state)
    
    km_model_pretrain.fit(data_with_target)
    km_model = KMeans(n_clusters=self.k,
                      init=km_model_pretrain.cluster_centers_[:,:35],
                      n_init=1,
                      max_iter=1)
    
    km_model.fit(X)

    self.km_model = km_model
    self.cluster_centers_ = km_model.cluster_centers_
    return self

  def transform(self, X, y=None):
      clusters = self.km_model.predict(X)
      return clusters[:,np.newaxis]
      
  def fit_transform(self, X, y=None):
      self.fit(X, y)
      return self.transform(X, y)






def Get_engineered_features_and_forest(df,method="corr"):
    #method = "corr" first eliminates highly correlated features and then tunes the Forest's hyperparameters
    #method = "rfe" first tunes the Forest's hyperparameters with all data, and then eliminates features recursively to optimize its performance

    # ----------------- STEP 0 - GET TRAINING AND TEST DATA ---------------------------
    df.sample(frac=1).reset_index(drop=True) #shuffle data & don't create a new column with original indices
    df.drop("client_id",inplace=True,axis=1)
    X_train_all, y_train_all = df.loc[:, df.columns != 'target'],df.loc[:,'target']



    #---------------- OPTIONAL STEP - REMOVE HIGHLY CORRELATED (AND CONSTANT) FEATURES -------------------------
    if(method=="corr"):
        
        # Define steps
        # step1 = {'Constant Features': {'frac_constant_values': 0.95}} #remove features of which X% of clients report the same thing
        step2 = {'Correlated Features': {'correlation_threshold': 0.8}} #Remove features that have more than X% correlation

        # Place steps in a list in the order you want them execute it
        steps = [step2]

        # Initialize FeatureSelector()
        fs = FeatureSelector()

        # Apply feature selection methods in the order they appear in steps
        fs.fit(X_train_all, y_train_all, steps)

        # Keep only the selected features
        X_train_all = fs.transform(X_train_all)

  
    #split of training set
    X_train, y_train = X_train_all[0:int(0.9*len(X_train_all))],y_train_all[0:int(0.9*len(y_train_all))] #first 90% of rows
    X_test,y_test = X_train_all[int(0.9*len(X_train_all)):],y_train_all[int(0.9*len(y_train_all)):] #last 10%

    print("Training set shape: {}\n".format(X_train.shape))
    print("Test set shape: {}\n".format(X_test.shape))


    #----------------------- STEP 2 -TUNE FOREST HYPERPARAMETERS  ------------
    # Initiate classifier instance
    estimator = RandomForestClassifier(random_state=42)

    #get class weights
    classes = np.unique(y_train)
    class_weights_arr = sklearn.utils.class_weight.compute_class_weight(class_weight = 'balanced', classes = classes, y = y_train)

    class_weights_dict = [{0:class_weights_arr[0]},{1:class_weights_arr[1]}] #gridsearchCV wants it as a list

    # Define parameter grid
    param_grid = { 'n_estimators': [200],
                  'class_weight': [None, 'balanced'],
                  'max_features': ['auto', 'sqrt', 'log2'],
                  'max_depth' : [3, 4, 5, 6, 7, 8],
                  'min_samples_split': [0.005, 0.01, 0.05, 0.10],
                  'min_samples_leaf': [0.005, 0.01, 0.05, 0.10],
                  'criterion' :['gini', 'entropy']     ,
                  'n_jobs': [-1],
                  'class_weight': class_weights_dict
                  }

    # Initialize GridSearch object
    gscv = GridSearchCV(estimator, param_grid, cv = 5,  n_jobs= -1, verbose = 1, scoring = 'precision' ) #Optimize for precision

    print("\nStarting the hyperparameter search for Random Forest now")
    # Fit gscv
    gscv.fit(X_train, y_train)

    # Get best parameters and score
    best_params = gscv.best_params_
    best_score = gscv.best_score_
          
    # Update classifier parameters
    estimator.set_params(**best_params)

    # Fit classifier
    estimator.fit(X_train, y_train)

    # Make predictions
    y_pred_train = estimator.predict(X_train)
    y_pred_test = estimator.predict(X_test)

    # Measure performance
    accuracy_train = metrics.accuracy_score(y_train, y_pred_train)
    accuracy_test = metrics.accuracy_score(y_test, y_pred_test)

    # Message to user
    print(f'The accuracy of the classifier on the train set was: {accuracy_train*100}')
    print(f'The accuracy of the classifier on the test set was: {accuracy_test*100}')



    #---------------- OPTIONAL STEP - FIND MOST RELEVANT FEATURES BY ITERATIVELY TRAINING THE TUNED MODEL -------------------------
    if(method=="rfe"):
        # Initiate classifier instance
        estimator = RandomForestClassifier(random_state = 42)

        # Update classifier parameters
        estimator.set_params(**best_params)

        # Define steps
        step1 = {'RFECV Features': {'cv': 5,
                                    'estimator': estimator,
                                    'step': 1,
                                    'scoring': 'precision',
                                    'verbose': 50}}

        # Place steps in a list in the order you want them execute it
        steps = [step1]

        # Initialize FeatureSelector()
        fs = FeatureSelector()

        # Apply feature selection methods in the order they appear in steps
        fs.fit(X_train_all, y_train_all, steps)

        # Get selected features
        X_train_all = fs.transform(X_train_all)

        #override earlier split
        X_train, y_train = X_train_all[0:int(0.9*len(X_train_all))],y_train_all[0:int(0.9*len(y_train_all))] #first 90% of rows
        X_test,y_test = X_train_all[int(0.9*len(X_train_all)):],y_train_all[int(0.9*len(y_train_all)):] #last 10%

    print("Training set shape: {}\n".format(X_train.shape))
    print("Test set shape: {}\n".format(X_test.shape))

    return X_train,y_train, X_test, y_test, estimator
