#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 18:52:13 2022

Train:

This script trains different models using different window sizes and epochs.
Results are saved in 'Results/'


@author: curro
"""
import os
import csv
import math
import argparse
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
from model_utils import configure_callbacks
from sklearn.metrics import mean_squared_error, mean_absolute_error


from plot_utils import plot_series, plot_predictions, plot_training, plot_gtruth_and_predictions

from data_handler import load_dataset
from model_utils import build_model, predict

def parser_opt():
    parser = argparse.ArgumentParser(description="Train script")
    parser.add_argument('-d','--dataset', type=str, required=False, default='ACN_data', help='Select dataset')
    parser.add_argument('-m','--model_name', type=str, required=True, help='Models available: CNN_LSTM,Simple_RNN,LSTM,LSTM_stacked,Bidirectional_LSTM,Simple_ANN')
    parser.add_argument('-w','--window_size', type=int, required=True, help='Input sequence length, 96 or 168 values (4-7 days)')
    parser.add_argument('-e','--epochs', type=int, required=True, help = 'Number of training epochs')
    parser.add_argument('-b','--batch_size', type=int, required=False, default = 16, help = 'Batch size for training' )

    opt = parser.parse_args()
    return opt

def main(opt):
    # Input args
    dataset = opt.dataset 
    model_name = opt.model_name 
    window_size = opt.window_size
    epochs = opt.epochs
    batch_size = opt.batch_size
    
    # Load and prepare the dataset
    series,times = load_dataset(dataset)

    # Plot some data
    plt.figure(figsize=(10, 6))
    plot_series(times, series, start=0, end = 24 * 7)
    plt.title('Data used sample')

    # Normalize the dataset
    scaler = MinMaxScaler(feature_range=(0.1, 1)) 
    series_scaled = scaler.fit_transform(series)

    # Split training and validation sets
    split_time = int(len(times) * 0.66)

    time_train = times[:split_time]
    x_train = series[:split_time] # Not scaled
    x_train_scaled = series_scaled[:split_time]

    time_test = times[split_time:]
    x_test = series[split_time:] # Not scaled
    x_test_scaled = series_scaled[split_time:]

    # Create the data generators for training
    num_features = 1
    train_generator = TimeseriesGenerator(x_train_scaled,
                                        x_train_scaled,
                                        length=window_size,
                                        batch_size=batch_size)
    validation_generator = TimeseriesGenerator(x_test_scaled,
                                            x_test_scaled,
                                            length=window_size,
                                            batch_size=batch_size)

    # Delete previous session to obtain static results
    tf.keras.backend.clear_session()
    tf.random.set_seed(51)
    np.random.seed(51)

    # Build the model:
    model = build_model(model_name,
                    window_size,
                    num_features)

    # Print model summary
    model.summary()

    # Configure callbacks
    callbacks = configure_callbacks(early_stopping = True,patience=10, 
                                    checkpoint= False, model_name= model_name )

    # Compilate the model  
    optimizer = 'adam'
    metrics = ['acc']
    model.compile(loss=tf.keras.losses.Huber(),
                optimizer=optimizer,
                metrics=metrics)

    # Train model
    history = model.fit_generator(generator=train_generator,
                                verbose=1,
                                epochs=epochs,
                                validation_data=validation_generator,
                                callbacks=callbacks)

    # Create new dir to save results
    try:
        os.mkdir(f'Results/train_{model_name}_ws{window_size}_epochs{epochs}_results')
    except:
        pass

    # Save the trained model 
    model.save(f'Results/train_{model_name}_ws{window_size}_epochs{epochs}_results/{model_name}_ws{window_size}_epochs{epochs}_model.h5')

    # Plot the training and validation accuracy and loss at each epoch
    plot_training(history)
    plt.savefig(f'Results/train_{model_name}_ws{window_size}_epochs{epochs}_results/history.png')


    # Run predictions on training and validation set
    train_predict = predict(model,
                            train_generator,
                            scaler)

    test_predict = predict(model,
                        validation_generator,
                        scaler)

    # Validate the model using RMSE and MAE
    trainScore = math.sqrt(mean_squared_error(x_train[window_size:], train_predict[:,0]))
    testScore = math.sqrt(mean_squared_error(x_test[window_size:], test_predict[:,0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    print('Test Score: %.2f RMSE' % (testScore))

    train_mae = mean_absolute_error(x_train[window_size:], train_predict[:,0]) 
    test_mae = mean_absolute_error(x_test[window_size:], test_predict[:,0])
    print('Train Score: %.2f MAE' % (train_mae))
    print('Test Score: %.2f MAE' % (test_mae))

    # Write validation results on csv
    csv_file = open(f'Results/train_{model_name}_ws{window_size}_epochs{epochs}_results/validation_metrics.txt', 'w')
    csv_writer = csv.writer(csv_file, delimiter = '\n')
    csv_writer.writerow(['Train Score: %.2f RMSE' % (trainScore), 'Test Score: %.2f RMSE' % (testScore)])
    csv_writer.writerow(['Train Score: %.2f MAE' % (train_mae), 'Test Score: %.2f MAE' % (test_mae)])
    csv_file.close()

    # Plot train and test predictions
    plot_predictions(train_predict,
                    test_predict,
                    window_size,
                    times,
                    series,
                    x_train)

    # Plot train and test predictions separately
    plt.figure()
    plt.plot(x_train[window_size:])
    plt.plot(train_predict[:,0])
    plt.title('Predictions on train')
    plt.savefig(f'Results/train_{model_name}_ws{window_size}_epochs{epochs}_results/pred_train.png')

    plt.figure()
    plt.plot(x_test[window_size:])
    plt.plot(test_predict[:,0])
    plt.title('Predictions on test')
    plt.savefig(f'Results/train_{model_name}_ws{window_size}_epochs{epochs}_results/pred_test.png')


    # Plot train and test with zoom
    plot_gtruth_and_predictions(x_train, train_predict, window_size, start = 0, end = 168)
    plt.title('Predictions on train (zoomed sample)')
    plt.savefig(f'Results/train_{model_name}_ws{window_size}_epochs{epochs}_results/pred_train_zoom.png')

    plot_gtruth_and_predictions(x_test, test_predict, window_size, start = 0, end = 168)
    plt.title('Predictions on test (zoomed sample)')
    plt.savefig(f'Results/train_{model_name}_ws{window_size}_epochs{epochs}_results/pred_test_zoom.png')


if __name__ =='__main__':
    opt = parser_opt()
    main(opt)
