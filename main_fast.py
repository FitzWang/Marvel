# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 20:31:41 2021

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
    if len(inputPara) == 1:
        inputPara = int(inputPara[0])
    else:
        print("Input should be one integer")
        raise IOError
    ####### define parameters
    nPointSuborder = 500
    loglik_nearMax = False
    #######
    spec = Marvel_fast.Spectrum()
    gpr = Marvel_fast.GPR()
    # Generate GP fit
    pathSpectrum = spec.pathbase / '..' / 'spectrum'
    spectrumList = os.listdir(pathSpectrum)
    spectrumList.sort()
    # Create GP output target folder
    if not os.path.exists(spec.pathbase / '..' /'GPFitted'):
        try: os.makedirs(spec.pathbase / '..' / 'GPFitted')
        except FileExistsError:
            pass
    # Create result folder
    datetimeNow = str(datetime.datetime.now()).split(" ")[0].replace('-','_')
    pathresult = Path(os.getcwd()) / '..' /'results' / datetimeNow
    if not os.path.exists(pathresult):
        try: os.makedirs(pathresult)
        except FileExistsError:
            pass
    # GP fit
    RV1Value = spectrumList[inputPara].split('.')[0].split('_')[-1]
    spectrum = pd.read_csv(pathSpectrum / spectrumList[inputPara])
    suborders,nSplit = spec.spectrumToSuborder(spectrum,nPointSuborder)
    GPFitted1 = gpr.PredOneGP(suborders)
    GPFitted1.to_csv(spec.pathbase / '..' /'GPFitted' / spectrumList[inputPara],index = False)
    
    while True:
        GPFittedList = os.listdir(spec.pathbase / '..' / 'GPFitted')
        timeSum = 0
        if len(GPFittedList) == len(spectrumList):
            print('Waiting time: {}'.format(timeSum))
            break
        elif timeSum >= 600:
            print('Timeout for waiting other cores to fit GP!')
            raise RuntimeError
        else:
            time.sleep(2)
            timeSum += 2
    # GPFittedList = os.listdir(spec.pathbase / '..' / 'GPFitted')       
    
    # Loop for files in GPFitted folder
    GPFittedList.sort()
    for i in range(len(GPFittedList)):       
        GPFitted2 = pd.read_csv(spec.pathbase / '..' / 'GPFitted'/ GPFittedList[i])
        RV2Value = GPFittedList[i].split('.')[0].split('_')[-1]
        length1 = len(GPFitted1)
        length2 = len(GPFitted2)
        index1 = GPFitted1.loc[GPFitted1['separation']==True].index.values
        index1 = np.append(index1,length1)
        index2 = GPFitted2.loc[GPFitted2['separation']==True].index.values
        index2 = np.append(index2,length2)
        if len(index1) != len(index2):
            print("WARNING! SIZE OF TWO INDECES ARE NOT SAME")
        
        # Create empty data array
        noSuborders = len(index1)-1
        RunningTime = np.empty(noSuborders, dtype=float)
        RVout = np.empty(noSuborders, dtype=float)
        std = np.empty(noSuborders, dtype=float)
        Order = np.repeat([i for i in range(len(nSplit))],nSplit)
        # Loop for suborders
        for j in range(noSuborders):
            timeStart = time.perf_counter()
            result = gpr.optimizeRV(GPFitted1.iloc[index1[j]:index1[j+1]],GPFitted2.iloc[index2[j]:index2[j+1]],loglik_nearMax)
            duration = time.perf_counter()-timeStart
            
            # fill in data
            RunningTime[j] = duration
            if loglik_nearMax == False:
                RVout[j] = result.x.item()
                std[j] = 1
            else:
                RVout[j] = result[0].x.item()
                loglikmean = result[1][round(len(rv[1])/2)]
                std[j] = np.sqrt(np.sum((np.array(result[1]) - loglikmean)**2)/len(result[1]))
            # print('RV:{}, time{}'.format(result.x.item(),duration))
        dfResults = pd.DataFrame({'RunningTime':RunningTime,'RVout':RVout,'std':std,'Order':Order})
        resultsName = 'output' + str(inputPara) + '_' + str(i) + '_' + str(int(RV1Value) - int(RV2Value)) + '.csv'
        dfResults.to_csv(pathresult / resultsName, index = False)
