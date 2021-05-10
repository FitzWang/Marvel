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
import re
        
if __name__ == '__main__':
    inputPara = sys.argv[1:]
    lenPara = len(inputPara)
    if len(inputPara) == 1:
        inputPara = int(inputPara[0])
    else:
        print("Input should be one integer")
        raise IOError
    ####### define parameters
    loglik_nearMax = True
    #######
    GPfitidx= pd.read_csv("data_opt.csv")
    GPfitidxSub = GPfitidx.iloc[101*inputPara:101*(inputPara+1)]
    
    spec = Marvel_fast.Spectrum()
    gpr = Marvel_fast.GPR()
    
    # Create result folder
    datetimeNow = str(datetime.datetime.now()).split(" ")[0].replace('-','_')
    pathresult = Path(os.getcwd()) / '..' /'results' / datetimeNow
    if not os.path.exists(pathresult):
        try: os.makedirs(pathresult)
        except FileExistsError:
            pass
    LmatrixList = os.listdir(spec.pathbase /'..'/'Python'/'Lmatrix')
    LmatrixList.sort(key=lambda f: int(re.sub('\D', '', f)))
    
    GPFittedList = os.listdir(spec.pathbase / '..' / 'GPFitted')
    GPFittedList.sort()
    
    #load all Lmatrix
    LmatrixFull = []
    for jj in range(len(LmatrixList)):       
        Ltmp = np.load(spec.pathbase/'Lmatrix'/LmatrixList[jj])
        LmatrixFull.append(Ltmp)
    
    for ii in range(len(GPfitidxSub)):
        inputPara1 = GPfitidxSub['index1'].iloc[ii]
        inputPara2 = GPfitidxSub['index2'].iloc[ii]
        GPFitted1 = pd.read_csv(spec.pathbase / '..' / 'GPFitted'/ GPFittedList[inputPara1])
        RV1Value = '.'.join(GPFittedList[inputPara1].split('.')[0:-1]).split('_')[-1]   
        GPFitted2 = pd.read_csv(spec.pathbase / '..' / 'GPFitted'/ GPFittedList[inputPara2])
        RV2Value = '.'.join(GPFittedList[inputPara2].split('.')[0:-1]).split('_')[-1]
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
        nSplit[nSplit==0]=1
        noSuborders = len(index1)-1
        RunningTime = np.empty(noSuborders, dtype=float)
        RVout = np.empty(noSuborders, dtype=float)
        RVMean = np.empty(noSuborders, dtype=float)
        var = np.empty(noSuborders, dtype=float)
        Order = np.repeat([i for i in range(len(nSplit))],nSplit)
        # Loop for suborders
        RVsum = 0
        for j in range(noSuborders):
            timeStart = time.perf_counter()
            Ls = LmatrixFull[j]
            result = gpr.optimizeRVQA_Lsaved(GPFitted1.iloc[index1[j]:index1[j+1]],GPFitted2.iloc[index2[j]:index2[j+1]],Ls,loglik_nearMax)
            duration = time.perf_counter()-timeStart
            # fill in data
            RunningTime[j] = duration
            if loglik_nearMax == False:
                RVsum += result
                RVMean[j] = RVsum/(j+1)
                RVout[j] = result
                var[j] = 1
            else:
                RVsum += result[0]
                RVout[j] = result[0]
                RVMean[j] = RVsum/(j+1)
                var[j] = result[1]
            # print('RV:{},var:{}, time{}'.format(result[0],result[1],duration))
        dfResults = pd.DataFrame({'RunningTime':RunningTime,'RV':RVout,'var':var,'RVmean':RVMean,'Order':Order})
        resultsName = 'output' + str(int(inputPara1/10)) + str(int(inputPara1%10)) + '_' + str(int(inputPara2/10)) +\
            str(int(inputPara2%10)) + '_' + str(round(float(RV1Value) - float(RV2Value),2)) + '.csv'
        dfResults.to_csv(pathresult / resultsName, index = False)
        ## check if result is surely written into disk
        for trytime in range(5):
            newList = os.listdir(pathresult)
            if resultsName not in newList:
                time.sleep(1)
                dfResults.to_csv(pathresult / resultsName, index = False)
            else:
                break
        print(resultsName + 'Extra trytimes: {}'.format(trytime))
        
        # ## check if need to delete nSplit file
        # lengthSpectra = len(GPFittedList)
        # if inputPara1 == (lengthSpectra-2) and inputPara2 == (lengthSpectra-1):
        #     os.remove('nSplit.csv')