# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 16:48:32 2023

@author: YUNDAM
"""

import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np
import pymannkendall as mk
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from yellowbrick.cluster.elbow import kelbow_visualizer

class monthlyEnergyBenchmarking():
    
    def __init__(self, energyData, energyColumn):
        
        self.energyData = energyData
        self.energyColumn = energyColumn
        self.year = ['2019', '2020', '2021']
        self.month = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']  
            
        
    def decomposeEnergy(self):
        
        '''
        
        Time-series decomposition to indentify seasonal, trend, and residual components in aggregated energy data
        
        '''
        
        # self.energyData = energyData
        
        # generate new columns for decomposed components        
        decompositionList = ['season', 'trend', 'resid']

        self.decompositionColumnList = []
        for a in decompositionList:
            for b in self.year:
                for c in self.month:
                    self.decompositionColumnList.append(a + '_' + b + c)
        
        
        self.dataDecomposition = pd.DataFrame(0, index=np.arange(len(self.energyData)), columns = self.decompositionColumnList)
        
        # decompose aggregated monthly energy data into seasonal, trend, and residual components
        for k in range(len(self.energyData)):
            
            tsData = self.energyData.loc[k, self.energyColumn] / self.energyData.loc[:,'gross_floor_area'][k]
            decomposed = seasonal_decompose(tsData, model='additive', period=12, extrapolate_trend = 'freq')
            self.dataDecomposition.iloc[k, :] = np.concatenate( (decomposed.seasonal, decomposed.trend, decomposed.resid) )


        self.dataDecomposition.index = self.energyData.index
        self.dataDecomposition = pd.concat([self.energyData, self.dataDecomposition], axis = 1)
        
        return self.dataDecomposition
    
    def mannKendall(self, dataDecomposition):
        
        '''
        
        Mann-Kendall test to quantify a trend in energy data and identy its direction (increasing, decreasing, or no trend)     
        
        '''
        
        self.dataMannKendall = dataDecomposition
        trendColumnList = [x for x in self.decompositionColumnList if "trend" in x]
        
        kendallTau = []
        pValue = []
        trendType = []
        
        for k in range(len(self.dataMannKendall)):
            
            tsData = self.dataMannKendall[trendColumnList].iloc[k,:] # Based on the decomposed trend data
            trendKendall = mk.original_test(tsData)
            trendType.append(trendKendall[0])
            kendallTau.append(trendKendall[4])
            pValue.append(trendKendall[2])
        
        self.dataMannKendall['trend_type'] = trendType
        self.dataMannKendall['p_value'] = pValue
        self.dataMannKendall['kendall_tau'] = kendallTau
        
        return self.dataMannKendall
    
    def kMeans(self, dataDecomposition):
        
        '''     
        
        Time-series clustering to group buildings with similar operation patterns
        
        '''
        
        self.dataClustering = dataDecomposition
        
        seasonColumnList = [x for x in self.decompositionColumnList if "season" in x]
        
        tsValue = self.dataDecomposition[seasonColumnList].values
        tsValue = np.nan_to_num(tsValue)

        scaler = TimeSeriesScalerMeanVariance(mu=0, std=1) # z-normailzaion
        dataScaled = scaler.fit_transform(tsValue)
        dataScaled = np.nan_to_num(dataScaled)
        
        dataScaled = np.resize(dataScaled, (len(dataScaled), 12))

        # Find the optimal number of clusters
        kmeansModel = TimeSeriesKMeans(random_state=0)
        visualizer = kelbow_visualizer(kmeansModel, dataScaled, k=(2,10))
        numberClusters = visualizer.elbow_value_

        # Fit a K-means model
        kmeansModelEnergy = TimeSeriesKMeans(n_clusters = numberClusters, metric="euclidean", verbose=True, random_state=0)
        energyCluster = kmeansModelEnergy.fit_predict(dataScaled)

        # Save the centroids
        energyCenter = np.zeros((numberClusters, 12))

        for i in range(numberClusters):
            
            energyCenter[i, :] = kmeansModel.cluster_centers_[i].ravel()

        np.savetxt("energyCentroid.csv", energyCenter, delimiter=",") # centroid
        
        self.dataClustering['energy_cluster'] = energyCluster
        
        return self.dataClustering