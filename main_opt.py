# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 00:48:20 2021

@author: guang
"""

import Marvel_fast
import numpy as np
import pandas as pd
import os
import time
import sys
import datetime
from pathlib import Path

        
if __name__ == '__main__':
    inputPara = sys.argv[1:]
    lenPara = len(inputPara)
    if len(inputPara) == 2:
        inputPara1 = int(inputPara[0])
        inputPara2 = int(inputPara[1])
    else:
        print("Input should be one integer")
        raise IOError
    ####### define parameters
    loglik_nearMax = False
    #######
    
    spec = Marvel_fast.Spectrum()
    gpr = Marvel_fast.GPR()
    
    # Create result folder
    datetimeNow = str(datetime.datetime.now()).split(" ")[0].replace('-','_')
    pathresult = Path(os.getcwd()) / '..' /'results' / datetimeNow
    if not os.path.exists(pathresult):
        try: os.makedirs(pathresult)
        except FileExistsError:
            pass
    
    GPFittedList = os.listdir(spec.pathbase / '..' / 'GPFitted')
    GPFittedList.sort()
    
    GPFitted1 = pd.read_csv(spec.pathbase / '..' / 'GPFitted'/ GPFittedList[inputPara1])
    RV1Value = GPFittedList[inputPara1].split('.')[0].split('_')[-1]   
    GPFitted2 = pd.read_csv(spec.pathbase / '..' / 'GPFitted'/ GPFittedList[inputPara2])
    RV2Value = GPFittedList[inputPara2].split('.')[0].split('_')[-1]
    length1 = len(GPFitted1)
    length2 = len(GPFitted2)
    index1 = GPFitted1.loc[GPFitted1['separation']==True].index.values
    index1 = np.append(index1,length1)
    index2 = GPFitted2.loc[GPFitted2['separation']==True].index.values
    index2 = np.append(index2,length2)   
    if len(index1) != len(index2):
        print("WARNING! SIZE OF TWO INDECES ARE NOT SAME")
        
    # Create empty data array
    nSplit = pd.read_csv('nSplit.csv')['nSplit'].ravel()
    noSuborders = len(index1)-1
    RunningTime = np.empty(noSuborders, dtype=float)
    RVout = np.empty(noSuborders, dtype=float)
    RVMean = np.empty(noSuborders, dtype=float)
    std = np.empty(noSuborders, dtype=float)
    Order = np.repeat([i for i in range(len(nSplit))],nSplit)
    # Loop for suborders
    RVsum = 0
    for j in range(noSuborders):
        timeStart = time.perf_counter()
        result = gpr.optimizeRV(GPFitted1.iloc[index1[j]:index1[j+1]],GPFitted2.iloc[index2[j]:index2[j+1]],loglik_nearMax)
        duration = time.perf_counter()-timeStart
        # fill in data
        RunningTime[j] = duration
        if loglik_nearMax == False:
            RVsum += result.x.item()
            RVMean[j] = RVsum/(j+1)
            RVout[j] = result.x.item()
            std[j] = 1
        else:
            RVsum += result.x.item()
            RVout[j] = result[0].x.item()
            RVMean[j] = RVsum/(j+1)
            loglikmean = result[1][round(len(rv[1])/2)]
            std[j] = np.sqrt(np.sum((np.array(result[1]) - loglikmean)**2)/len(result[1]))
        # print('RV:{}, time{}'.format(result.x.item(),duration))
    dfResults = pd.DataFrame({'RunningTime':RunningTime,'RV':RVout,'std':std,'RVmean':RVMean,'Order':Order})
    resultsName = 'output' + str(int(inputPara1/10)) + '_' + str(int(inputPara2%10)) + '_' + str(int(RV1Value) - int(RV2Value)) + '.csv'
    dfResults.to_csv(pathresult / resultsName, index = False)
    
    
    ## check if need to delete nSplit file
    lengthSpectra = len(GPFittedList)
    if inputPara1 == (lengthSpectra-2) and inputPara2 == (lengthSpectra-1):
        os.remove(nSplit.csv)