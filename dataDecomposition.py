# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 09:28:03 2023

@author: Yun-Dam
"""

import os
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np

import pymannkendall as mk
from typing import Tuple, Union

os.chdir(r'C:\Users\YUNDAM\Desktop\benchmarking-clustering-cpr')

# read the data
data = pd.read_csv('building_energy_2019_2021_preprocessed.csv', index_col=0, encoding='cp949')

# %% energy data

year = ['2019', '2020', '2021']
month = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']

elecColumnList = []
for a in year:
    for b in month:
        elecColumnList.append('elec' + a + b)

gasColumnList = []
for a in year:
    for b in month:
        gasColumnList.append('gas' + a + b)

elec = data.loc[:, elecColumnList] 
gas = data.loc[:,gasColumnList]

# %% decompose time-series energy data

decompositionList = ['season', 'trend', 'resid']

elecDecomposeColumnList = []
for a in decompositionList:
    for b in year:
        for c in month:
            elecDecomposeColumnList.append('elec' + b + c + '_' + a)

gasDecomposeColumnList = []
for a in decompositionList:
    for b in year:
        for c in month:
            gasDecomposeColumnList.append('gas' + b + c + '_' + a)

dataDecompositionElec = pd.DataFrame(0, index=np.arange(len(elec)), columns = elecDecomposeColumnList)
dataDecompositionGas = pd.DataFrame(0, index=np.arange(len(gas)), columns = gasDecomposeColumnList)

for k in range(len(elec)):
    tsData = elec.iloc[k,:] / data.loc[:,'연면적(㎡)'][k]
    decomposed = seasonal_decompose(tsData, model='additive', period=12)
    dataDecompositionElec.iloc[k, :] = np.concatenate( (decomposed.seasonal, decomposed.trend, decomposed.resid) )

for k in range(len(gas)):
    tsData = gas.iloc[k,:] / data.loc[:,'연면적(㎡)'][k]
    decomposed = seasonal_decompose(tsData, model='additive', period=12)
    dataDecompositionGas.iloc[k, :] = np.concatenate( (decomposed.seasonal, decomposed.trend, decomposed.resid) )

dataDecompositionElec.index = data.index
dataDecompositionGas.index = data.index

dataDecomposition = pd.concat([dataDecompositionElec, dataDecompositionGas], axis = 1)
dataDecomposition = pd.concat([data, dataDecomposition], axis = 1)


# %% Mann-Kendall test to quantify a trend in time-series data

kendallTau = []
pValue = []
trendType = []

for k in range(len(elec)):
    
    tsData = elec.iloc[k,:] / data.loc[:,'연면적(㎡)'][k]
    trendKendall = mk.original_test(tsData)
    trendType.append(trendKendall[0])
    kendallTau.append(trendKendall[4])
    pValue.append(trendKendall[2])

dataDecomposition['trend_type'] = trendType
dataDecomposition['p_value'] = pValue
dataDecomposition['kendall_tau'] = kendallTau


# Save data
dataDecomposition.to_csv('./building_energy_2019_2021_decomposed.csv', encoding='cp949')  


# %%
