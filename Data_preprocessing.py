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
from Helper_functions  import *
from model_factory import *
from automated_feature_engineering import *




#Data cleaner function
def clean_minmax(data,transform="normal",nan_handler="drop"): #Nan handler can be set to "drop" or "to_zero", transform to "normal" , "robust" or "power"
  #------------------ STEP 1.1: HANDLE THE NAN VALUES -----------------------

  if(nan_handler == "drop"):
    #Education seems to be MNAR! Much less churners have reported their education...
    printMissingData(data)
    #create feature indicating if education was reported (can be informative)
    data['has_reported_education'] = data['customer_education'].notnull().astype('int')
    #drop the columns with more than 25% missing values
    data.drop(['customer_education','customer_children','customer_relationship'],axis=1,inplace=True)
    #Remove all other rows containing NaN values (up to 4% loss in data)
    data_len = data.shape[0]
    data = data.dropna()
    print('\nData loss after dropping NaNs: {}%'.format((data.shape[0]-data_len)/data_len*100))

  elif(nan_handler == "to_zero"):
    printMissingData(data)
    #create feature indicating if education was reported (can be informative)
    data['has_reported_education'] = data['customer_education'].notnull().astype('int')
    #drop the columns with more than 25% missing values
    data.drop(['customer_education','customer_children','customer_relationship'],axis=1,inplace=True)
    #Fill nan data values with dataset's creation date
    data['customer_since_all'] = data['customer_since_all'].fillna(dt.datetime(2018, 6,1).strftime("%Y-%m")) #CAN BE PROBLEM IF YOUNGER PEOPLE MORE LIKELY TO CHURN!
    data['customer_since_bank'] = data['customer_since_bank'].fillna(dt.datetime(2018, 6,1).strftime("%Y-%m"))
    #Fill remaining nan values with 0s
    data.fillna(0)


  #------------------ STEP 1.2: HANDLE DATE TYPE FEATURES -----------------------
  dataset_creation_date = dt.datetime(2018, 6,1).strftime("%Y-%m")
  # today = dt.date.today().strftime("%Y-%m")


  #Turn dates into whole numbers (months passed since today)
  data["customer_since_all"] = [get_age_in_months(x,dataset_creation_date) for x in data["customer_since_all"]]
  data["customer_since_bank"] = [get_age_in_months(x,dataset_creation_date) for x in data["customer_since_bank"]]
  data["customer_birth_date"] = [get_age_in_months(x,dataset_creation_date) for x in data["customer_birth_date"]]


  #------------------ STEP 1.3: Transform the categorical variables -----------------------
  #Change POSTAL CODES to one-hot encoded regions
  Brussels,Flanders,Wallonia,Other = getRegionClass(data['customer_postal_code'])
  data['brussels_postal_code'] = Brussels
  data['flanders_postal_code'] = Flanders
  data['wallonia_postal_code'] = Wallonia
  data['other_postal_code'] = Other
  data.drop(["customer_postal_code"], axis=1,inplace=True)

  #Change OCCUPATION CODES to one-hot encoded regions. DROP FOR NOW SINCE DISTRIBUTIONS BETWEEN CHURNERS AND CONTROL GROUP SEEM NEGLIGIBLE
  data.drop(["customer_occupation_code"], axis=1,inplace=True)


  #------------------ STEP 1.4: General operations -----------------------
  #turn visits into integers
  data["visits_distinct_so"] = data["visits_distinct_so"].astype(int)
  data["visits_distinct_so_areas"] = data["visits_distinct_so_areas"].astype(int)

  #Turn gender into 0-1 instead of 1-2 encoding
  def gender_to_binary(x):
    if(x==1):
      return 0
    else:
      return 1
  data["customer_gender"] = [gender_to_binary(x) for x in data["customer_gender"]]


  #------------------ STEP 1.5: Normalization -----------------------
  cols_to_normalize = ['bal_insurance_21',"bal_insurance_23","cap_life_insurance_fixed_cap","cap_life_insurance_decreasing_cap","prem_fire_car_other_insurance",
                      "bal_personal_loan","bal_mortgage_loan","bal_current_account","bal_pension_saving","bal_savings_account","bal_savings_account_starter",
                      "bal_current_account_starter","visits_distinct_so","visits_distinct_so_areas","customer_since_all","customer_since_bank","customer_birth_date"]

  if(transform == "normal"):
    for x in cols_to_normalize:
      data[x] = normalize_minmax(data[x])
    return data

  elif (transform == "robust"):
    trans = RobustScaler()
    data[cols_to_normalize] = trans.fit_transform(data[cols_to_normalize])
    return data

  elif(transform == "power"): #For yeo power transform
    from sklearn.preprocessing import power_transform
    data[cols_to_normalize] = power_transform(data[cols_to_normalize],method='yeo-johnson') #Can handle full dataframes
    return data

  elif(transform==None):
    return data




# --- Define parent functions for data cleaning --- (Normally data is also saved to pass on to feature engineering and model notebooks)
def get_clean_train_data(transform,nan_handler):
  train_data1 = pd.read_csv('data_original/train_month_1.csv')
  train_data1 = clean_minmax(train_data1,transform=transform,nan_handler=nan_handler)

  train_data2 = pd.read_csv('data_original/train_month_2.csv')
  train_data2 = clean_minmax(train_data2,transform=transform,nan_handler=nan_handler)

  train_data3 = pd.read_csv('data_original/train_month_3_with_target.csv')
  train_data3 = clean_minmax(train_data3,transform=transform,nan_handler=nan_handler)

  return train_data1, train_data2, train_data3


def get_clean_test_data(transform,nan_handler):
  test_data1 = pd.read_csv('data_original/test_month_1.csv')
  test_data1 = clean_minmax(test_data1,transform=transform,nan_handler=nan_handler)

  test_data2 = pd.read_csv('data_original/test_month_2.csv')
  test_data2 = clean_minmax(test_data2,transform=transform,nan_handler=nan_handler)

  test_data3 = pd.read_csv('data_original/test_month_3.csv')
  test_data3 = clean_minmax(test_data3,transform=transform,nan_handler=nan_handler)

  return test_data1, test_data2, test_data3





#Function to get dataframe with manually engineered features (month3 - month1)
def get_manual_features_df(data,transform,nan_handler,filename):
  #Get the original data
  if(data=="train"):
    df1 = pd.read_csv('data_original/train_month_1.csv')
    df1 = clean_minmax(df1,transform=None,nan_handler=nan_handler)
    df1.sort_values(by=['client_id'],inplace=True) #Sort values for consistent row positions per customer across dataframes
    df3 = pd.read_csv('data_original/train_month_3_with_target.csv')
    df3 = clean_minmax(df3,transform=None,nan_handler=nan_handler)
    df3.sort_values(by=['client_id'],inplace=True) #Sort values for consistent row positions per customer across dataframes

  if(data=="test"):
    df1 = pd.read_csv('data_original/test_month_1.csv')
    df1 = clean_minmax(df1,transform=None,nan_handler=nan_handler)
    df1.sort_values(by=['client_id'],inplace=True) #Sort values for consistent row positions per customer across dataframes
    df3 = pd.read_csv('data_original/test_month_3.csv')
    df3 = clean_minmax(df3,transform=None,nan_handler=nan_handler)
    df3.sort_values(by=['client_id'],inplace=True) #Sort values for consistent row positions per customer across dataframes
  
  
  
  #create df to hold transformed dataset 
  new_df = pd.DataFrame({"client_id":df1['client_id']})
  #define new features
  new_df["homebanking_change"] = df3["has_homebanking"] - df1["has_homebanking"] #0 = nothing changed, 1 = activated it, -1 = deactivated it
  new_df["insurance_21_change"] = df3["has_insurance_21"] - df1["has_insurance_21"]
  new_df["insurance_23_change"] = df3["has_insurance_23"] - df1["has_insurance_23"]
  new_df["life_ins_fixed_cap_change"] = df3["has_life_insurance_fixed_cap"] - df1["has_life_insurance_fixed_cap"]
  new_df["life_ins_decreas_cap_change"] = df3["has_life_insurance_decreasing_cap"] - df1["has_life_insurance_decreasing_cap"]
  new_df["fire_insurance_change"] = df3["has_fire_car_other_insurance"] - df1["has_fire_car_other_insurance"]
  new_df["personal_loan_change"] = df3["has_personal_loan"] - df1["has_personal_loan"]
  new_df["mortgage_loan_change"] = df3["has_mortgage_loan"] - df1["has_mortgage_loan"]
  new_df["current_acc_change"] = df3["has_current_account"] - df1["has_current_account"]
  new_df["pension_savings_acc_change"] = df3["has_pension_saving"] - df1["has_pension_saving"]
  new_df["savings_acc_change"] = df3["has_savings_account"] - df1["has_savings_account"]
  new_df["savings_acc_starter_change"] = df3["has_savings_account_starter"] - df1["has_savings_account_starter"]
  new_df["current_acc_starter_change"] = df3["has_current_account_starter"] - df1["has_current_account_starter"]
  new_df["bal_change_insurance_21"] = df3["bal_insurance_21"] - df1["bal_insurance_21"]
  new_df["bal_change_insurance_23"] = df3["bal_insurance_23"] - df1["bal_insurance_23"]
  new_df["cap_life_ins_fixed"] = df3["cap_life_insurance_fixed_cap"] - df1["cap_life_insurance_fixed_cap"]
  new_df["cap_life_ins_decreasing"] = df3["cap_life_insurance_decreasing_cap"] - df1["cap_life_insurance_decreasing_cap"]
  new_df["fire_premium_change"] = df3["prem_fire_car_other_insurance"] - df1["prem_fire_car_other_insurance"]
  new_df["bal_change_personal_loan"] = df3["bal_personal_loan"] - df1["bal_personal_loan"]
  new_df["bal_change_mortgage_loan"] = df3["bal_mortgage_loan"] - df1["bal_mortgage_loan"]
  new_df["bal_change_current_account"] = df3["bal_current_account"] - df1["bal_current_account"]
  new_df["bal_change_pension_savings"] = df3["bal_pension_saving"] - df1["bal_pension_saving"]
  new_df["bal_change_savings"] = df3["bal_savings_account"] - df1["bal_savings_account"]
  new_df["bal_change_savings_starter"] = df3["bal_savings_account_starter"] - df1["bal_savings_account_starter"]
  new_df["bal_change_current_starter"] = df3["bal_current_account_starter"] - df1["bal_current_account_starter"]
  new_df["visits_change"] = df3["visits_distinct_so"] - df1["visits_distinct_so"]
  #SOME STATISTICS EXPECTED TO BE IMPORTANT STATICALLY:
  new_df["customer_since_all"] = df3["customer_since_all"]
  new_df["customer_gender"] = df3["customer_gender"]
  new_df["customer_birth_date"] = df3["customer_birth_date"]
  new_df["customer_self_employed"] = df3["customer_self_employed"] 
  new_df["has_reported_education"] = df3["has_reported_education"] 
  new_df["brussels_postal_code"] = df3["brussels_postal_code"] 
  new_df["flanders_postal_code"] = df3["flanders_postal_code"] 
  new_df["wallonia_postal_code"] = df3["wallonia_postal_code"]
  new_df["other_postal_code"] = df3["other_postal_code"] 
  #If training set then add target
  if(data=="train"):
      new_df["target"] = df3["target"]


  cols_to_normalize = ['bal_change_insurance_21',"bal_change_insurance_23","cap_life_ins_fixed","cap_life_ins_decreasing","fire_premium_change",
                      "bal_change_personal_loan","bal_change_mortgage_loan","bal_change_current_account","bal_change_pension_savings","bal_change_savings","bal_change_savings_starter",
                      "bal_change_current_starter","visits_change","customer_since_all","customer_birth_date"]

  # Normalize the data
   if(transform == "normal"):
     for x in cols_to_normalize:
       new_df[x] = normalize_minmax(new_df[x])

   elif(transform == "robust"):
     trans = RobustScaler()
     new_df[cols_to_normalize] = trans.fit_transform(new_df[cols_to_normalize])

   elif(transform == "power"): #For yeo power transform
     from sklearn.preprocessing import power_transform
     new_df[cols_to_normalize] = power_transform(new_df[cols_to_normalize],method='yeo-johnson') #Can handle full dataframes


  return new_df




#Get input data for the LSTM model
def get_combined_lstm_data(data="train",transform="robust",nan_handler="drop"):
    if(data =="train"):
        df1, df2, df3 = get_clean_train_data(transform=transform,nan_handler=nan_handler)
    elif(data =="test"):
        df1, df2, df3 = get_clean_test_data(transform=transform,nan_handler=nan_handler)
            
        
    df1.sort_values(by=['client_id'],inplace=True) #Sort values for consistent row positions per customer across dataframes
    df1.drop(["client_id"],inplace=True,axis=1) #Drop client id
    df1.reset_index(drop=True,inplace=True)

    
    df2.sort_values(by=['client_id'],inplace=True) #Sort values for consistent row positions per customer across dataframes
    df2.drop(["client_id"],inplace=True,axis=1) #Drop client id
    df2.reset_index(drop=True,inplace=True)
    
    df3.sort_values(by=['client_id'],inplace=True) #Sort values for consistent row positions per customer across dataframes
    y_train_lstm = df3["target"]
    df3.drop(["client_id","target"],inplace=True,axis=1) #Drop client id
    df3.reset_index(drop=True,inplace=True)
    
    
    #create arrays to accumulate datasets
    X_lstm = np.empty((df1.shape[0],3,df1.shape[1]),dtype='float')
    
    #ACCUMULATE TRAINING DF
    #Add all observations from train_df1
    for x in range(len(df1)):
      X_lstm[x][0] = df1.iloc[x,:]
      #Add all observations from train_df2
    for x in range(len(df2)):
      X_lstm[x][1] = df2.iloc[x,:]
    #Add all observations from train_df3
    for x in range(len(df3)):
      X_lstm[x][2] = df3.iloc[x,:]
    
    print("LSTM input data shape: {}".format(X_lstm.shape))


