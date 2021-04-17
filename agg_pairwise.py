# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 16:52:32 2021

@author: guang
"""

import pandas as pd
import os
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import time

def ExtendOutput(pathbase,csvList):
    for outFile in csvList:
        dataFrame = pd.read_csv(pathbase/outFile)
        dataFrame['RV'] = dataFrame['RV'].mul(-1)
        dataFrame['RVmean'] = dataFrame['RVmean'].mul(-1)
        name = outFile.split('.')[0].split('_')
        index1 = name[0].split('output')[1]
        index2 = name[1]
        relativeRV = str(-int(name[2]))
        newFileName = 'output' + index2 + '_' +  index1 + '_' + relativeRV + '.csv'
        dataFrame.to_csv(pathbase/newFileName,index = False)
        
def ReadAll(N,pathbase,csvList):
    resultAll = []
    for i in range(N):
        outputTemp = []
        for j in range(N):
            index1 = str(int(i/10)) + str(i%10)
            index2 = str(int(j/10)) + str(j%10)
            filename = [item for item in csvList if item.startswith('output'+index1+'_'+index2)]
            assert len(filename) <= 1, 'May there are duplicates files with same index'
            assert len(filename) >= 1, 'No output named start with this index:{},{}'.format(index1,index2)
            dataFrame = pd.read_csv(pathbase/filename[0])
            outputTemp.append(dataFrame)
        resultAll.append(outputTemp)
    return resultAll

def RVSuborder(resultList,suborderLen):
    RVList = []
    for idx1 in range(len(resultList)):
        start = time.perf_counter()
        dfsubList = resultList[idx1]
        RVI = np.empty(suborderLen,dtype=float) # for all suborders
        VarI = np.empty(suborderLen,dtype=float)
        for suborderIdx in range(suborderLen):
            RVtmp = np.empty(N,dtype=float)
            vartmp = np.empty(N,dtype=float)
            for idx2 in range(len(dfsubList)):
                subdf = dfsubList[idx2].iloc[[suborderIdx]]                
                RVtmp[idx2] = subdf['RV'].item()
                vartmp[idx2] = subdf['var'].item()
            weighti = (1/vartmp)/np.sum(1/vartmp)
            RVi = np.sum(RVtmp*weighti)
            # under assumption of independent estimator
            vari = 1/np.sum(1/vartmp)
            RVI[suborderIdx] = RVi
            VarI[suborderIdx] = vari
        dfRVI = pd.DataFrame({"RVI":RVI,"VarI":VarI})
        RVList.append(dfRVI)
        print(time.perf_counter()-start)
    return RVList

def RVAggAll(RVList):
    RVout = np.empty(len(RVList),dtype=float)
    Varout = np.empty(len(RVList),dtype=float)
    for i in range(len(RVList)):
        df = RVList[i]
        weight = (1/df['VarI'])/(1/df['VarI']).sum()
        RV = (df["RVI"]*weight).sum()
        var = 1/(1/df['VarI']).sum()
        RVout[i] = RV
        Varout[i] = var
    return RVout,Varout

## First compute all suborders, then pairwise
def RVAggAll2(resultList):
    RV  = np.empty(len(resultList),dtype=float)
    Var = np.empty(len(resultList),dtype=float)
    for i in range(len(resultList)):
        subList = resultList[i]
        RVtmp = np.empty(len(subList),dtype=float)
        Vartmp = np.empty(len(subList),dtype=float)
        for j in range(len(subList)):
            dataFrame = subList[j]
            weight = 1/dataFrame['var']
            weightSum = sum(weight)
            weightedRV = sum(dataFrame['RV']*weight/weightSum)
            weightedVar = 1/weightSum
            RVtmp[j] = weightedRV
            Vartmp[j] = weightedVar
        weight = 1/Vartmp
        weightSum = sum(weight)
        weightedRV = sum(weight*RVtmp/weightSum)
        weightedVar = 1/weightSum
        RV[i] = weightedRV
        Var[i] = weightedVar
    return RV,Var
if __name__ == '__main__':
    #############parameters########
    fileExtend = False  # specify if need to extend output file (e.g. only optimize pairwise 1-0, not 0-1)
    saveImg = False
    targetFolder = 'HPC_2021_03_22'
    cutOffOrder = 43
    N = 41
    #############################
    pathbase = Path(os.getcwd()) / '..' /'results'/ targetFolder
    
    csvList = os.listdir(pathbase)
    csvList = [ filename for filename in csvList if filename.endswith('.csv')]
    # extend the output file
    if fileExtend == True:
        ExtendOutput(pathbase,csvList) 
        if not os.path.exists(pathbase / 'graph'):  
            os.makedirs(pathbase / 'graph')
        csvList = os.listdir(pathbase)
        csvList = [ filename for filename in csvList if filename.endswith('.csv')]
        csvList.sort()
        
    # Read all files into one 2d array    
    resultAll = ReadAll(N,pathbase,csvList)
    lengthOut = len(resultAll[0][0])
    # Deal with suborders rv
    RVList = RVSuborder(resultAll,lengthOut)
    # Final output
    RVout,Varout = RVAggAll(RVList)
    
    # RVout,Varout = RVAggAll2(resultAll)
    
    # plot the results
    RVTrue = pd.read_csv(pathbase/'RVTrue.txt')['RVTrue'].to_numpy()
    fig, ax = plt.subplots(figsize=(18,9))
    X = np.arange(N,dtype=int)
    meansubTrueRV = RVTrue-np.mean(RVTrue)
    
    ax.plot(X,meansubTrueRV,'o',label = "True Radial Velocity (mean-substracted)")
    turerms1 = np.around(np.sqrt(((RVout-meansubTrueRV)**2).mean()),decimals=2)
    ax.plot(X,RVout,'o-',label = "Total Suborders, RMSE:"+str(turerms1))
    ax.fill_between(X, (RVout + 1.96*np.sqrt(Varout)), (RVout - 1.96*np.sqrt(Varout)), alpha=0.3) 
    ax.set_xlabel("Index of epochs")
    ax.set_ylabel("Radial Velocity/ m/s (mean-substracted)")
    ax.grid(True, color='gainsboro', linestyle='-', linewidth=0.5)
    fig.legend()
    
    fig1, ax1 = plt.subplots(figsize=(11,7))
    error = 1.96*np.sqrt(Varout)
    ax1.plot(X,meansubTrueRV,'-',label = "True Radial Velocity (mean-substracted)")
    turerms1 = np.around(np.sqrt(((RVout-meansubTrueRV)**2).mean()),decimals=2)
    ax1.errorbar(X, RVout, yerr=error,color='black',capsize=3, fmt='o',label = "Estimated Radial Velocity, RMSE:"+str(turerms1))
    ax1.set_xlabel("Index of epochs")
    ax1.set_ylabel("Radial Velocity/ m/s (mean-substracted)")
    ax1.grid(True, color='gainsboro', linestyle='-', linewidth=0.5)
    fig1.legend()
    
    if saveImg == True:
        fig.savefig(pathbase / 'graph' / "RVcomparison.png")
        fig1.savefig(pathbase / 'graph' / "RVcomparison1.png")