# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 00:36:18 2021

@author: guang
"""

import pandas as pd
import sys

if __name__ == '__main__':
    
    inputPara = sys.argv[1:]
    lenPara = len(inputPara)
    if len(inputPara) == 1:
        inputPara = int(inputPara[0])
    else:
        print("Input should be one integer")
        raise IOError
    
    GPFitIndex = [i for i in range(inputPara)]
    pd.DataFrame({'index':GPFitIndex}).to_csv('data_gp.csv',index=False)
    
    optindex1 = []
    optindex2 = []
    for i in range(inputPara):
        for j in range(i,inputPara):
            optindex1.append(i)
            optindex2.append(j)
    pd.DataFrame({'index1':optindex1,'index2':optindex2}).to_csv('data_opt.csv',index=False)