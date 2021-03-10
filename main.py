# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 19:58:44 2021

@author: guang
"""

import Marvel
import Marvel_torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from utilities import *
import time
import sys
import os
import datetime


def simpleTest(torch = False):   
    marvel = Marvel.Spectrum(velocityShift=200)
    marvel1 = Marvel.Spectrum(velocityShift=250)
    subSpectrum,noPoints = marvel.subSpectrum(lowerWavl = 500, upperWavl = 501)
    subSpectrum1,noPoints1 = marvel1.subSpectrum(lowerWavl = 500, upperWavl = 501)        
    allSuborders = marvel.spectrumToSuborder()
    allSuborders1 = marvel1.spectrumToSuborder()
    if torch == False:
        gpr = Marvel.GPR()
    else:
        gpr = Marvel_torch.GPR()
    len60 = len(allSuborders[20])
    for i in range(len60):
        startTime = time.perf_counter()
        rv = gpr.optimizeRV(allSuborders[20][i],allSuborders1[20][i],cuda=True, loglik_nearMax=True)
        loglikmean = rv[1][round(len(rv[1])/2)]     
        loglikstd = np.sqrt(np.sum((np.array(rv[1]) - loglikmean)**2)/len(rv[1]))
        print((time.perf_counter() - startTime),rv[0],loglikstd)

def generateSpec(Vmag = 9.,exposureTime = 900.):
    os.chdir('..')
    pathbase = os.getcwd() + '\\spectrum'
    os.chdir('Python')
    # creat spectrum when folder is empty
    if len(os.listdir(pathbase)) == 0:
        jitter = 150*np.sin(np.pi*np.arange(0,1.5,0.15))
        RVBase = 500
        allSuborders = []
        RV = RVBase + jitter
        for i in range(len(jitter)):
            velocityShift = int(RV[i])
            R, telescopeArea, effWavelSpectrograph, samplingResolution, throughputInterpolator, lambdaMin, lambdaMax = getSpectrograph("Marvel")
            
            # Choose the stellar parameters. Cf Data/ folder to see what choice there is.        
            Teff = 5700          # Effective temperature  [K]
            logg = 4.5           # logarithm of the gravity 
            vsini = 2            # A measure for the stellar rotation, determines the width of the spectral lines
            vmicro = 1           # Micro-turbulent velocity, affects the depth of your spectral lines
            FeH = 0.0            # Metallicity: how much metals in the stellar atmosphere. Determines the depth of the spectral lines
            Vmag = Vmag            # Magnitude of the star (i.e. its brightness)         
            exposureTime = exposureTime        
            # Load and process the stellar spectrum
            # This is for 1 telescope only. Use 4*telescopeArea instead of telescopeArea if the spectrum for 4 telescopes is needed.        
            spectrum = loadSpectrum2(Teff, logg, vmicro, FeH, vsini, R, effWavelSpectrograph, samplingResolution, velocityShift)
            spectrum = electronFlux(spectrum, telescopeArea, throughputInterpolator, lambdaMin, lambdaMax, exposureTime, Vmag, 
                                    includePoissonNoise=True, normalizeSpectrum=True)
            spectrum.to_csv(pathbase + "\\Spectrum" + str(i) + '_' + str(velocityShift) +".csv")

def loadSpectrum():
    # get spectrum storage folder
    os.chdir('..')
    pathbase = os.getcwd() + '\\spectrum\\'
    os.chdir('Python')
    spectrumList = []
    RVlist = []
    spectrumName = os.listdir(pathbase)
    for name in spectrumName:
        marvel = Marvel.Spectrum(path = pathbase + name,nPointSuborder=450)
        allSuborders = marvel.spectrumToSuborder()
        spectrumList.append(allSuborders)
        RVlist.append(name.split(".")[0].split("_")[-1])
    return spectrumList,RVlist

def computeRV(spectrum1,spectrum2,filename,relativeRV,torch = False, cuda = False,loglik_nearMax = False):
    sumDuration = 0
    sumRV = 0
    count = 0
    for i in range(len(spectrum1)):
        for j in range(len(spectrum1[i])):
            count += 1
            subspectrum1 = spectrum1[i][j]
            subspectrum2 = spectrum2[i][j]           
            startTime = time.perf_counter() 
            if torch == False:
                gpr = Marvel.GPR()
            else:
                gpr = Marvel_torch.GPR()
            rv = gpr.optimizeRV(subspectrum1,subspectrum2, cuda = cuda, loglik_nearMax = loglik_nearMax)
            duration = time.perf_counter() - startTime
            sumDuration += duration
            if loglik_nearMax == False:
                rvOut = rv.x.item()
                std = 1
            else:
                rvOut = rv[0]
                loglikmean = rv[1][round(len(rv[1])/2)]
                std = np.sqrt(np.sum((np.array(rv[1]) - loglikmean)**2)/len(rv[1]))
            sumRV += rvOut
            aveNow = np.around(sumRV/count,decimals=2)
            print(duration,rvOut,relativeRV,aveNow)           
            outFile = open(filename, "a")
            outFile.write(str(duration) + "," + str(rvOut) + "," + str(std) +"," + str(aveNow) + "," + str(i) + "," + str(j) + "\n")
            outFile.close()
            
if __name__ == "__main__":
    generateSpec()
    spectrumList,RVlist = loadSpectrum()
    lenSinSpectrum = len(spectrumList)
    # creat new result folder
    datetimeNow = str(datetime.datetime.now()).split(" ")[0]
    os.chdir('..')
    pathbase = os.getcwd() + '\\results\\' + datetimeNow 
    if not os.path.exists(pathbase):
        os.chdir('results')
        os.makedirs(datetimeNow)
        os.chdir('..')
        os.chdir('Python')
    for i in range(lenSinSpectrum):
        for j in range(i+1,lenSinSpectrum):
            spectrum1 = spectrumList[i]
            spectrum2 = spectrumList[j]
            relativeRV = int(RVlist[i]) -  int(RVlist[j])
            filename = pathbase + "\\output" + str(i) + "_" + str(j) + "_" + str(relativeRV) + ".csv"
            outFile = open(filename, "w")
            outFile.write("RunningTime,RV,var,AveraeRVNow,Order,SubOrder\n")
            outFile.close()
            computeRV(spectrum1,spectrum2,filename,relativeRV,torch = False, cuda = False,loglik_nearMax = True)

