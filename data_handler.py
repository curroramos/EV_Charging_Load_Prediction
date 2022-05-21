#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 18:55:31 2022

Data handler:
    
Function used to load different datasets

@author: curro
"""

import pandas as pd


def load_dataset(name):
    if name == 'ACN_data':
        return get_ACN()
    elif name == 'XXX':
        return get_XXX()

def get_ACN():
    # Load the dataset
    file_path = "data_preprocessing/processed_data/data2019.json"
    df = pd.read_json(file_path)
    # New dataframe with training data - 1 column
    times = df.filter(['index'])
    df_for_training = df.filter(['val_total_power']).astype(float)

    #Convert pandas dataframe to numpy array
    series = df_for_training.values
    series = series.astype('float64') #Convert values to float
    
    return series,times

def get_XXX():
    # not implemented
    pass
    