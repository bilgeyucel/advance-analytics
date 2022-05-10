# -*- coding: utf-8 -*-
"""
Created on Tue May 10 09:34:12 2022

@author: tijl_
"""

# ---------------- HYPERPARAMETER TUNING - HYPERBAND ALGORITHM ------------------
#SOURCE: https://towardsdatascience.com/hyperparameter-tuning-with-kerastuner-and-tensorflow-c4a4d690b31a

# !pip install keras-tuner 
import keras_tuner as kt
import tensorflow_addons as tfa

#get base input shape
print(X_train.shape)
in_shp = list(X_train.shape[1:]) #47 dimensions
print(in_shp)
#input_nn = tensorflow.keras.Input(shape = (1,400)) #2D input for Convolutional layers
epochs = 20


def build_model(hp):
    """
    Builds model and sets up hyperparameter space to search. (For small dense network)
    
    Parameters
    ----------
    hp : HyperParameter object
        Configures hyperparameters to tune.
        
    Returns
    -------
    model : keras model
        Compiled model with hyperparameters to tune.
    """
    # Initialize sequential API and start building model.
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=in_shp))
    # Tune the number of hidden layers and units in each.
    # Number of hidden layers: 1 - 5
    # Number of Units: 32 - 256 with stepsize of 32
    for i in range(1, hp.Int("num_layers", 2, 6)):
        model.add(
            tf.keras.layers.Dense(
                units=hp.Int("units_" + str(i), min_value=32, max_value=256, step=32),
                activation="relu")
            )
        
        # Tune dropout layer with values from 0 - 0.3 with stepsize of 0.1.
        model.add(tf.keras.layers.Dropout(hp.Float("dropout_" + str(i), 0, 0.3, step=0.1)))
    
    # Add output layer.
    model.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))
    
    # Tune learning rate for Adam optimizer with values from 0.01, 0.001, or 0.0001
    hp_learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
    
    # Define optimizer, loss, and metrics
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=["accuracy",tfa.metrics.F1Score()])
    
    return model

# Instantiate the tuner. HyperBand algorithm used for its effectiveness

#Factor and max_epochs determine how many random models our search starts off to consider (log3(20+1)) = 3. So every iteration
#Considers 3 models in this case. It recognizes the base search space by permutations of the hp() objects defined in build_model
tuner = kt.Hyperband(build_model,
                     kt.Objective("val_f1", direction="max"), #documentation on flexible objective setting: https://keras.io/guides/keras_tuner/getting_started/
                     max_epochs=epochs,
                     factor=3, 
                     hyperband_iterations=10)
                     #directory="kt_dir", #directory to save progress to
                     #project_name="kt_hyperband") #projectname = subdirectory under the main directory


# Display search space summary
tuner.search_space_summary()

#tuner.search method takes similar input to mode.l.fit()
stop_early = tf.keras.callbacks.EarlyStopping(monitor=tfa.metrics.F1Score(), patience=5)
tuner.search(X_train, y_train, epochs=epochs, validation_split=0.2, callbacks=[stop_early], verbose=2)



# ------------------------------ DEEP FEATURE SYNTHESIS (#https://www.featuretools.com/) ---------------------
# DID NOT GET IT WORKING

import featuretools as ft
es = ft.EntitySet(id="training_data")

#When we do not initialize Woodwork, the first columns of dataframes added to the EntitySet will be taken as the index. Since featuretools is built for one-to-many
#relationships, it always matches the index of the parent dataframe to the key_column of the child. So below transformations are needed for featuretools to work
train_data1.rename(mapper={'client_id': 'client_id_ind'}, axis=1,inplace=True)
train_data1["client_id_col"] = train_data1["client_id_ind"]
train_data2.rename(mapper={'client_id': 'client_id_ind'}, axis=1,inplace=True)
train_data2["client_id_col"] = train_data2["client_id_ind"]
train_data3.rename(mapper={'client_id': 'client_id_ind'}, axis=1,inplace=True)
train_data3["client_id_col"] = train_data3["client_id_ind"]

es["train1"] = train_data1
es["train2"] = train_data2
es["train3"] = train_data3

es = es.add_relationship("train2", "client_id_ind", "train1", "client_id_col") #parent_df name, parent_key, child_df name, child_key (assumes one-to-many by default)
es = es.add_relationship("train3", "client_id_ind", "train2", "client_id_col")
es.plot()

#Extend this for time series analysis: https://featuretools.alteryx.com/en/stable/guides/time_series.html NOT SURE IF IT IS WORTH THE EFFORT FOR US


