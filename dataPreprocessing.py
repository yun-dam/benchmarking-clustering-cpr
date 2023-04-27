# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 11:52:53 2023

@author: Yun-Dam
"""


# import plotly.express as px
import pandas as pd
import os

# os.chdir(r'C:\Users\USER\Desktop\benchmarking-clustering-cpr')


# Read the data
data = pd.read_csv('building_energy_2019_2021.csv', index_col=0, encoding='cp949')


# %% generate column lists
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

# %% Drop missing values in the specific columns

data = data.dropna(subset = ['연면적(㎡)', '주_용도_코드', '사용승인_일', 'PNU', '주_용도_코드_명'] + elecColumnList + gasColumnList)


# Drop use types with few samples

data["주_용도_코드_명"] = data["주_용도_코드_명"].astype("category") # Convert to category
useTypeList = [x for x in data.주_용도_코드_명.unique().tolist() if x not in ["제2종근린생활시설", "제1종근린생활시설", "교육연구시설", "노유자시설", "숙박시설",
               "업무시설", "종교시설", "공장", "의료시설", "근린생활시설"]]


for k in useTypeList:
    data = data[data.주_용도_코드_명 != k]

# Add the EUI columns
data['EUI2019']=data.loc[:, [x for x in elecColumnList if '2019' in x] + [y for y in gasColumnList if '2019' in y]].sum(axis=1) / data.loc[:,'연면적(㎡)']
data['EUI2020']=data.loc[:, [x for x in elecColumnList if '2020' in x] + [y for y in gasColumnList if '2020' in y]].sum(axis=1) / data.loc[:,'연면적(㎡)']
data['EUI2021']=data.loc[:, [x for x in elecColumnList if '2021' in x] + [y for y in gasColumnList if '2021' in y]].sum(axis=1) / data.loc[:,'연면적(㎡)']

# Remove outliers
euiCols = ['EUI2019', 'EUI2020', 'EUI2021', '연면적(㎡)'] # one or more

Q1 = data[euiCols].quantile(0.025)
Q3 = data[euiCols].quantile(0.975)
IQR = Q3 - Q1

data = data[~((data[euiCols] < (Q1 - 1.5 * IQR)) |(data[euiCols] > (Q3 + 1.5 * IQR))).any(axis=1)]

# Save data
data.to_csv('./building_energy_2019_2021_preprocessed.csv', encoding='cp949')  
