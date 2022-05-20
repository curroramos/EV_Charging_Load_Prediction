#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 08:05:30 2021

Data preprocessing:

Preprocess EV charging sessions raw data from https://ev.caltech.edu/dataset.
Creates a new file with the preprocessed data in directory './preprocessed_data/'


@author: curro
"""

import pandas as pd
import json 

# Read the data
data_path = 'raw_data/acndata_sessions_2019.json'

data = json.load(open(data_path))
df = pd.DataFrame(data["_items"])

# Convert string column to a datetime64 type
df['connectionTime'] = pd.to_datetime(df['connectionTime'])
df['disconnectTime'] = pd.to_datetime(df['disconnectTime'])

# We take the columns needed
df=df[['connectionTime','disconnectTime','kWhDelivered']]

# Add a column named dummy
df['dummy'] = 1

# Data time interval
timedelta = df['connectionTime'].max() - df['connectionTime'].min()

# Add a column for session time interval
interval = df['disconnectTime']-df['connectionTime'] 

# Change time format and compute the power
interval=interval.dt.total_seconds()
interval=interval/3600
df['sessionTime']=interval
df['instantPower']=df['kWhDelivered']/df['sessionTime']

# Create a datetimeindex
date_series=pd.date_range(start=df['connectionTime'].min(),end=df['disconnectTime'].max(),freq="1H")

# Create a dataframe with the datetimeindex and a column named dummy
date_df = pd.DataFrame(dict(date=date_series, dummy=1))

# Create an union, join the column dummy to df dataframe
cross_join = date_df.merge(df, on='dummy')
cond_join = cross_join[(cross_join.connectionTime <= cross_join.date) & (cross_join.date <= cross_join.disconnectTime)]
grp_join = cond_join.groupby(['date'])

# Create the new dataframe
final = (
    pd.DataFrame(dict(
        val_sessions=grp_join.size(),
        val_total_power=grp_join.instantPower.sum(),
        val_power_median=grp_join.instantPower.median()
    ), index=date_series)
    .fillna(0)
    .reset_index()
)

#%% Export 'final' dataframe into a pkl file
final.to_pickle("preprocessed_data/final_2018")


