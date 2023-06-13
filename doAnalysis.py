# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 09:28:03 2023

@author: Yun-Dam
"""

import pandas as pd
import monthlyEnergyBenchmarking

# read the data
data = pd.read_csv('./sampleData.csv', index_col=0, encoding='cp949')

# gernerate the energy column names
year = ['2019', '2020', '2021']
month = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']

elecColumnList = []
for a in year:
    for b in month:
        elecColumnList.append('elec' + a + b)

# use the module to conduct time-series decomposition, clustering, and Mann-Kendall test

meb = monthlyEnergyBenchmarking.monthlyEnergyBenchmarking(data, elecColumnList)

decomposedData = meb.decomposeEnergy()
mannkendallTestData = a.mannKendall(decomposedData)
clusteredData = a.kMeans(decomposedData)

