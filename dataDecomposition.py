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

os.chdir(r'C:\Users\USER\Desktop\benchmarking-clustering-cpr')

# read the data
data = pd.read_csv('building_energy_2019_2021_preprocessed.csv', index_col=0, encoding='cp949')


# %% 

elec = data.loc[:, ['elec201901',
'elec201902', 'elec201903', 'elec201904', 'elec201905', 'elec201906',
'elec201907', 'elec201908', 'elec201909', 'elec201910', 'elec201911',
'elec201912', 'elec202001', 'elec202002',
'elec202003', 'elec202004', 'elec202005', 'elec202006', 'elec202007',
'elec202008', 'elec202009', 'elec202010', 'elec202011', 'elec202012',
 'elec202101', 'elec202102', 'elec202103',
'elec202104', 'elec202105', 'elec202106', 'elec202107', 'elec202108',
'elec202109', 'elec202110', 'elec202111', 'elec202112' ]] 

gas = data.loc[:,['gas201901', 'gas201902', 'gas201903', 'gas201904',
'gas201905', 'gas201906', 'gas201907', 'gas201908', 'gas201909',
'gas201910', 'gas201911', 'gas201912', 'gas202001', 'gas202002', 'gas202003', 'gas202004', 'gas202005',
'gas202006', 'gas202007', 'gas202008', 'gas202009', 'gas202010',
'gas202011', 'gas202012', 'gas202101', 'gas202102', 'gas202103', 'gas202104', 'gas202105', 'gas202106',
'gas202107', 'gas202108', 'gas202109', 'gas202110', 'gas202111',
'gas202112']]

# %%
# 월별 데이터라서  period=12 로 잡음. 값 안넣어도 자동으로 12로 잡힘
# model="multiplicative" 넣으면 multiplicative decomposition 함
for k in range(50,100,1):
    tsData = elec.iloc[k,:] / data.loc[:,'연면적(㎡)'][k]
    
    dec = seasonal_decompose(tsData, model='additive', period=12)
    fig = dec.plot()
    fig.set_size_inches(9, 5)
    print(1-(np.var(dec.resid)/np.var(dec.trend+dec.resid)))
    
    
# %%
k = 411
tsData = elec.iloc[k,:] / data.loc[:,'연면적(㎡)'][k]

dec = seasonal_decompose(tsData, model='additive', period=12)
fig = dec.plot()
fig.set_size_inches(9, 5)
print(1-(np.var(dec.resid)/np.var(dec.trend+dec.resid)))

result = mk.original_test(tsData)
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
