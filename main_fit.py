# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 00:29:39 2021

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
    nPointSuborder = 900
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
    # GP fit
    spectrum = pd.read_csv(pathSpectrum / spectrumList[inputPara])
    suborders,nSplit = spec.spectrumToSuborder(spectrum,nPointSuborder)
    GPFitted = gpr.PredOneGP(suborders)
    GPFitted.to_csv(spec.pathbase / '..' /'GPFitted' / spectrumList[inputPara],index = False)
    # save suborder split information
    my_file = spec.pathbase/'nSplit.csv'
    if not my_file.is_file():
        pd.DataFrame({'nSplit':nSplit}).to_csv('nSplit.csv',index=False)