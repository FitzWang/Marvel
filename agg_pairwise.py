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
import copy

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

def SaveDFbySuborder(pathbase,resultList,suborderLen):
    pathTarget = pathbase / 'bysuborder'
    if not os.path.exists(pathTarget):
        os.makedirs(pathTarget)
    fileLength = len(os.listdir(pathTarget))
    if fileLength == suborderLen:
        print('Generated suborder files exist!')
    else:
        length = len(resultList)
        for i in range(suborderLen):
            outIdx1 = np.empty(length*length,dtype=int)
            outIdx2 = np.empty(length*length,dtype=int)
            RV = np.empty(length*length,dtype=float)
            Var = np.empty(length*length,dtype=float)
            for idx1 in range(length):
                for idx2 in range(length):
                    df = resultList[idx1][idx2]
                    index = idx1*length + idx2
                    outIdx1[index] = idx1
                    outIdx2[index] = idx2
                    RV[index] = df['RV'].iloc[i]
                    Var[index] = df['var'].iloc[i]
            dfOut = pd.DataFrame({'idx1':outIdx1,'idx2':outIdx2,'RV':RV,'var':Var})
            outFileName = 'suborder_'+str(i)+'.csv'
            dfOut.to_csv(pathTarget/outFileName,index=False)
    return pathTarget

def RVSuborder2(pathTarget,N):
    csvList = os.listdir(pathTarget)
    csvList = [filename for filename in csvList if filename.endswith('.csv') and filename.startswith('suborder')]
    csvList.sort()
    
    RVList = []
    for file in csvList:
        df = pd.read_csv(pathTarget/file)
        RVoneOrder = np.empty(N,dtype=float)
        VaroneOrder = np.empty(N,dtype=float)
        for idx1 in range(N):
            dfsub = df[df['idx1']==idx1]
            weight = (1/dfsub['var'])/(1/dfsub['var']).sum()
            weightedRV = (weight * dfsub['RV']).sum()
            weightedVar = 1/(1/dfsub['var']).sum()
            RVoneOrder[idx1] = weightedRV
            VaroneOrder[idx1] = weightedVar
        dfoneOrder = pd.DataFrame({'RVl':RVoneOrder,'varl':VaroneOrder})
        RVList.append(dfoneOrder)
    return RVList

def RVAggAll3(RVList,N):
    length = len(RVList)
    RVout = np.empty(N,dtype=float)
    Varout = np.empty(N,dtype=float)
    for idx1 in range(N):
        RVtmp = np.empty(length,dtype=float)
        Vartmp = np.empty(length,dtype=float)
        for i in range(length):
            RVtmp[i] = RVList[i]['RVl'].iloc[idx1]
            Vartmp[i] = RVList[i]['varl'].iloc[idx1]
        weight = 1/Vartmp
        weightsum = np.sum(weight)
        weightedRV = np.sum(RVtmp*weight/weightsum)
        weightedVar = 1/weightsum
        RVout[idx1] = weightedRV
        Varout[idx1] = weightedVar
    return RVout, Varout

def LeaveOneOutCorr(RVList,N,removeIdx):
    remainingList = copy.deepcopy(RVList)
    RVselected = remainingList.pop(removeIdx)
    RVRemaing,_ = RVAggAll3(remainingList,N)
    corr = np.corrcoef(RVRemaing, RVselected['RVl'])
    return corr[0,1]
    
if __name__ == '__main__':
    #############parameters########
    fileExtend = False  # specify if need to extend output file (e.g. only optimize pairwise 1-0, not 0-1)
    saveImg = False
    targetFolder = 'HPC_2021_03_22'
    cutOffOrder = 43
    N = 41
    ContaminationDet = False
    bottomK = 20
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
    print('Reading all the raw output CSV files......')
    start = time.perf_counter()
    resultAll = ReadAll(N,pathbase,csvList)
    print('time:{}s'.format(time.perf_counter()-start))
    
    lenSuborder = len(resultAll[0][0])
    start = time.perf_counter()
    print('Saving New CSV files by suborders......')
    pathDFByOrder = SaveDFbySuborder(pathbase,resultAll,lenSuborder)
    print('time:{}s'.format(time.perf_counter()-start))
    
    # Deal with suborders rv
    start = time.perf_counter()
    print('Computing RV estimation for all suborders......')
    RVListbySuborder = RVSuborder2(pathDFByOrder,N)
    # Final output
    RVout,Varout = RVAggAll3(RVListbySuborder,N)
    print('time:{}s'.format(time.perf_counter()-start))
    
    # compute correlation between using leave-one-out method
    if ContaminationDet:
        start = time.perf_counter()
        print('Computing correlation by leave-one-out method......')
        corr = map(lambda i:LeaveOneOutCorr(RVListbySuborder,N,i),range(lenSuborder))
        corr = list(corr)
        print('time:{}s'.format(time.perf_counter()-start))
        RemaingCorrIdx = np.array(corr).argsort()[bottomK:]
        RVListRmSmCorr = [RVListbySuborder[i] for i in RemaingCorrIdx]
        RVRmLowCorr,VarRmLowCorr = RVAggAll3(RVListRmSmCorr,N)
        
    
    # # Deal with suborders rv
    # RVList = RVSuborder(resultAll,lengthOut)
    # # Final output
    # RVout,Varout = RVAggAll(RVList)
    
    # # RVout,Varout = RVAggAll2(resultAll)
    
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
    ax1.set_title('RV estimation using all suborders')
    fig1.legend()
    
    if ContaminationDet:
        fig2, ax2 = plt.subplots(figsize=(11,7))
        error = 1.96*np.sqrt(VarRmLowCorr)
        ax2.plot(X,meansubTrueRV,'-',label = "True Radial Velocity (mean-substracted)")
        turerms1 = np.around(np.sqrt(((RVRmLowCorr-meansubTrueRV)**2).mean()),decimals=2)
        ax2.errorbar(X, RVRmLowCorr, yerr=error,color='black',capsize=3, fmt='o',label = "Estimated Radial Velocity, RMSE:"+str(turerms1))
        ax2.set_xlabel("Index of epochs")
        ax2.set_ylabel("Radial Velocity/ m/s (mean-substracted)")
        ax2.grid(True, color='gainsboro', linestyle='-', linewidth=0.5)
        ax2.set_title('RV estimation after removing suborders with low correlation')
        fig2.legend()
    
    # plot the histogram of variance using all suborders
    fig3, ax3 = plt.subplots(figsize=(11,7))
    ax3.hist(Varout)
    ax3.set_title('Histogram of Variance for Each RV Estimation')
    
    if saveImg == True:
        fig.savefig(pathbase / 'graph' / "RVcomparison.png")
        fig1.savefig(pathbase / 'graph' / "RVcomparison1.png")
        fig3.savefig(pathbase /'graph'/'hist.png')
        if ContaminationDet:
            fig2.savefig(pathbase / 'graph' / "RVcomparison2.png")