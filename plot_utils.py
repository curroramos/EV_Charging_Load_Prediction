#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 19:47:53 2022

Plot utils:
    
Functions used to represent time series, predictions, and training history

@author: curro
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Plot time series
def plot_series(time, series, format="-", start=0, end=None):
    plt.figure(figsize=(10, 6))
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)
    
# Plotting training loss and accuracy
def plot_training(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss, 'y', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    

# Plot train and test predictions
def plot_predictions(train_predict, test_predict, window_size, times, series, x_train ):
    # Shift train predictions for plotting:
    # We must shift the predictions so that they align on the x-axis with the original dataset. 
    trainPredictPlot = np.empty_like(series)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[window_size:len(x_train), :] = train_predict
    
    # Shift test predictions for plotting
    testPredictPlot = np.empty_like(series)
    testPredictPlot[:, :] = np.nan
    #testPredictPlot[len(trainPredict)+(seq_size*2)-1:len(dataset)-1, :] = testPredict
    testPredictPlot[len(x_train)+(window_size):len(series), :] = test_predict
    
    # Settings
    plt.style.use('seaborn')
    plt.rcParams["figure.figsize"] = (16, 8)
    pd.options.plotting.backend = "plotly"
    
    # Plot baseline and predictions
    # plt.plot(times, series)
    plt.plot(series)
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    plt.title('Predictions on the whole dataset')


    

# Plot real values and predicted from 'start' value to 'end' value
def plot_gtruth_and_predictions(x, predict, window_size, start, end):  
    plt.figure()
    plt.plot(x[window_size + start: window_size + end])
    plt.plot(predict[start:end,0])

    
    
        