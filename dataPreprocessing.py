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

# Drop missing values in the specific columns
data = data.dropna(subset = ['연면적(㎡)', '주_용도_코드', '사용승인_일', 'PNU', 
                    'elec201901', 'elec201902', 'elec201903',
                    'elec201904', 'elec201905', 'elec201906', 'elec201907', 'elec201908',
                    'elec201909', 'elec201910', 'elec201911', 'elec201912', 'gas201901',
                    'gas201902', 'gas201903', 'gas201904', 'gas201905', 'gas201906',
                    'gas201907', 'gas201908', 'gas201909', 'gas201910', 'gas201911',
                    'gas201912', 'elec202001', 'elec202002', 'elec202003', 'elec202004',
                    'elec202005', 'elec202006', 'elec202007', 'elec202008', 'elec202009',
                    'elec202010', 'elec202011', 'elec202012', 'gas202001', 'gas202002',
                    'gas202003', 'gas202004', 'gas202005', 'gas202006', 'gas202007',
                    'gas202008', 'gas202009', 'gas202010', 'gas202011', 'gas202012',
                    'elec202101', 'elec202102', 'elec202103', 'elec202104', 'elec202105',
                    'elec202106', 'elec202107', 'elec202108', 'elec202109', 'elec202110',
                    'elec202111', 'elec202112', 'gas202101', 'gas202102', 'gas202103',
                    'gas202104', 'gas202105', 'gas202106', 'gas202107', 'gas202108',
                    'gas202109', 'gas202110', 'gas202111', 'gas202112'])

# Drop use types with few samples

# data["주_용도_코드"] = data["주_용도_코드"].astype("category") # Convert to category
data["주_용도_코드_명"] = data["주_용도_코드_명"].astype("category") # Convert to category
# print(data["주_용도_코드_명"].value_counts())

useTypeList = [x for x in data.주_용도_코드_명.unique().tolist() if x not in ["제2종근린생활시설", "제1종근린생활시설", "교육연구시설", "노유자시설", "숙박시설",
               "업무시설", "종교시설", "공장", "의료시설", "근린생활시설"]]

for k in useTypeList:

    data = data[data.주_용도_코드_명 != k]


# Add the EUI columns

data['EUI2019']=data.loc[:,['elec201901', 'elec201902', 'elec201903', 'elec201904', 'elec201905',
       'elec201906', 'elec201907', 'elec201908', 'elec201909', 'elec201910',
       'elec201911', 'elec201912', 'gas201901', 'gas201902', 'gas201903',
       'gas201904', 'gas201905', 'gas201906', 'gas201907', 'gas201908',
       'gas201909', 'gas201910', 'gas201911', 'gas201912']].sum(axis=1) / data.loc[:,'연면적(㎡)']


data['EUI2020']=data.loc[:,['elec202001', 'elec202002',
       'elec202003', 'elec202004', 'elec202005', 'elec202006', 'elec202007',
       'elec202008', 'elec202009', 'elec202010', 'elec202011', 'elec202012',
       'gas202001', 'gas202002', 'gas202003', 'gas202004', 'gas202005',
       'gas202006', 'gas202007', 'gas202008', 'gas202009', 'gas202010',
       'gas202011', 'gas202012']].sum(axis=1) / data.loc[:,'연면적(㎡)']

data['EUI2021']=data.loc[:,['elec202101', 'elec202102', 'elec202103',
        'elec202104', 'elec202105', 'elec202106', 'elec202107', 'elec202108',
        'elec202109', 'elec202110', 'elec202111', 'elec202112', 'gas202101',
        'gas202102', 'gas202103', 'gas202104', 'gas202105', 'gas202106',
        'gas202107', 'gas202108', 'gas202109', 'gas202110', 'gas202111',
        'gas202112']].sum(axis=1) / data.loc[:,'연면적(㎡)']

# Remove outliers

euiCols = ['EUI2019', 'EUI2020', 'EUI2021', '연면적(㎡)'] # one or more

Q1 = data[euiCols].quantile(0.025)
Q3 = data[euiCols].quantile(0.975)
IQR = Q3 - Q1

data = data[~((data[euiCols] < (Q1 - 1.5 * IQR)) |(data[euiCols] > (Q3 + 1.5 * IQR))).any(axis=1)]

# Save data
data.to_csv('./building_energy_2019_2021_preprocessed.csv', encoding='cp949')  



