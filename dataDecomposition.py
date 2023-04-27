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

dataDecomposition = pd.concat([dataDecompositionElec, dataDecompositionGas], ignore_index=True, axis = 1)

# Save data
# data.to_csv('./building_energy_2019_2021_decomposed.csv', encoding='cp949')  

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
    
    
# %%



def trend(series: Union[np.ndarray, pd.Series, pd.DataFrame], 
          test: str ='consensus',
          tests: dict = {
            'hamed_rao': pymannkendall.hamed_rao_modification_test,
            'normal': pymannkendall.original_test,
            'yue_wang': pymannkendall.yue_wang_modification_test
           }) -> Tuple[list, float]:
    """
    Applies the Mann Kendall test  to the series and
    averages the results of each test in `tests`

    Parameters
    -----------
    series: Union[numpy.ndarray, pandas.Series, pandas.DataFrame]
        the input series - a univariate array
    test: str
        default: 'consensus'
    tests: dict
        a dict of pymannkendall tests. default tests are 
        `hamed_rao`, `normal` and `yue_wang`
    
    Returns
    --------
    Tuple[list, float]

    A list of result objects as well 
    as the trend [1=positive, 0=no trend, -1=negative]
    """
    trend = 0.
    if test == "consensus":
        results = []
        for t, v in tests.items():
            results.append(v(series))
        for r in results:
            if r.trend == 'decreasing':
                trend -= 1
            elif r.trend == 'increasing':
                trend += 1

    # consensus average
    return results, round(trend / len(tests))


a = trend(tsData)

result = mk.original_test(tsData)
