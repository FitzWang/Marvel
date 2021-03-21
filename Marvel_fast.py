# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 20:18:52 2021

@author: guang
"""
import GPy
import numpy as np
import os
import time
from utilities import *
import matplotlib.pyplot as plt
import pandas as pd
import scipy
from scipy import stats, signal
from scipy.optimize import minimize_scalar
from scipy import optimize
from scipy.spatial.distance import pdist, cdist, squareform
from pathlib import Path

class Spectrum():
    def __init__(self):
        self.pathbase = Path(os.getcwd())
    
    def CheckSpecExist(self, disp = True):
        if not os.path.exists(self.pathbase/'..'/'spectrum'):
            os.makedirs(self.pathbase/'..'/'spectrum')
        spectrumList = os.listdir(self.pathbase/'..'/'spectrum')
        if len(spectrumList) == 0:
            return False
        else:
            if disp == True:
                print('Existed spectra:')
                print(spectrumList)
            return True
        
    def GenOneSpectrum(self, velocityShift, Vmag = 9., exposureTime = 900.):
        R, telescopeArea, effWavelSpectrograph, samplingResolution, throughputInterpolator, lambdaMin, lambdaMax = getSpectrograph("Marvel")

        # Choose the stellar parameters. Cf Data/ folder to see what choice there is.
        
        Teff = 5700          # Effective temperature  [K]
        logg = 4.5           # logarithm of the gravity 
        vsini = 2            # A measure for the stellar rotation, determines the width of the spectral lines
        vmicro = 1           # Micro-turbulent velocity, affects the depth of your spectral lines
        FeH = 0.0            # Metallicity: how much metals in the stellar atmosphere. Determines the depth of the spectral lines
        Vmag = Vmag  ##11    # Magnitude of the star (i.e. its brightness) 
        
        exposureTime = exposureTime## 25*60
        
        # Load and process the stellar spectrum
        # This is for 1 telescope only. Use 4*telescopeArea instead of telescopeArea if the spectrum for 4 telescopes is needed.
        
        spectrum = loadSpectrum2(Teff, logg, vmicro, FeH, vsini, R, effWavelSpectrograph, samplingResolution, velocityShift)
        spectrum = electronFlux(spectrum, telescopeArea, throughputInterpolator, lambdaMin, lambdaMax, exposureTime, Vmag, 
                                includePoissonNoise=True, normalizeSpectrum=True)
        
        return spectrum
    
    def GetOrderIndex(self, spectrum):
        # Split spectrum to orders via trough of original signal
        miniInd = signal.argrelextrema(spectrum['NelectronsCont'].ravel(), np.less, order = 10)
        # remove the first split threshold point, since first order only contains few points
        miniInd = miniInd[0].tolist()[1:]
        return miniInd
    
    def GenSpectrum(self,velocity, Vmag = 9., exposureTime = 900., overwrite=False):
        def ApplyGen(velocity, Vmag, exposureTime):
            assert len(velocity.shape) == 1, 'Please input one-dim list or array'
            for i in range(len(velocity)):
                spectrum = self.GenOneSpectrum(velocity[i], Vmag, exposureTime)
                filename = 'Spectrum' + str(int(i/10)) + str(int(i%10)) + '_' + str(int(velocity[i])) + '.csv'
                spectrum.to_csv(self.pathbase/'..'/'spectrum'/filename,index=False)
                
        flag = self.CheckSpecExist(disp = False)
        if flag == False:
            ApplyGen(velocity, Vmag, exposureTime)
        else:
            if overwrite == True:
                import shutil
                print('Overwritting existing spectra!!!')
                shutil.rmtree(self.pathbase/'..'/'spectrum')
                os.makedirs(self.pathbase/'..'/'spectrum')
                ApplyGen(velocity, Vmag, exposureTime)
            else:
                print('Spectra exist, No need to generate.')
                
    def orderToSuborder(self,spectrum,nSplit):
        pointsPerOrder = int(np.ceil(len(spectrum)/nSplit))
        suborder = []
        for i in range(nSplit-1):
            suborder.append(spectrum.iloc[pointsPerOrder*i:pointsPerOrder*(1+i)])
        # for last sub-order, select obs to the end in case obs lost because of ceiling
        i = i+1
        suborder.append(spectrum.iloc[pointsPerOrder*i:])
        return suborder
    
    def spectrumToSuborder(self, spectrum, nPointSuborder):
        miniInd = self.GetOrderIndex(spectrum)
        spectrumByOrders = np.split(spectrum, miniInd, axis=0)
        nSplit = [round(len(i)/nPointSuborder) for i in spectrumByOrders]
        assert len(spectrumByOrders) == len(nSplit)
        suborders = []
        for i in range(len(nSplit)):
            if nSplit[i] == 0:
                suborders.append([spectrumByOrders[i]])
            else:
                suborders.append(self.orderToSuborder(spectrumByOrders[i],nSplit[i]))
        return suborders,nSplit
     
    def subSpectrum(self, spectrum, lowerWavl, upperWavl):
        subSpec = spectrum.loc[(spectrum["wavel"] > lowerWavl) & (spectrum["wavel"] < upperWavl)]
        return subSpec
    
    def plotSpectrum(self,spectrum,normalization = True):
        miniInd = signal.argrelextrema(spectrum['NelectronsCont'].ravel(), np.less, order = 10)
        fig, ax = plt.subplots(figsize=(18,9))
        X = spectrum['wavel']
        if normalization == True:
            Y = spectrum['NelectronsLine']
            Ylabel = "Flux [e-/exposure]"
        else:
            Y = spectrum['NelectronsCont']
            Ylabel = "Flux [original]"
        ax.plot(X, Y, c="steelblue")
        ax.set_xlabel("Wavelength [nm]")
        ax.set_ylabel(Ylabel)
        ax.grid(True, color='gainsboro', linestyle='-', linewidth=0.5)
        if normalization == False and len(miniInd[0])!=0:
            Xminimum = spectrum["wavel"].iloc[miniInd]
            Yminimum = spectrum["NelectronsCont"].iloc[miniInd]
            ax.plot(Xminimum, Yminimum, "or")
        plt.show()


class GPR():
    def __init__(self, h = 0.5, rho = 0.02):
        self.c = 299792458
        self.h = h
        self.rho = rho
        
    def regression(self,spectrum, holdPara = True):
        noPoints = len(spectrum)
        kernelMatern = GPy.kern.Matern52(input_dim=1,variance = self.h**2, lengthscale = self.rho)
        # Define data for regression
        X = spectrum["wavel"].ravel().reshape(noPoints,-1)
        Y = spectrum["NelectronsLine"].ravel().reshape(noPoints,-1)
        
        # poisson_likelihood = GPy.likelihoods.Poisson()
        # laplace_inf = GPy.inference.latent_function_inference.Laplace()
        # m = GPy.core.GP(X=X, Y=Y, likelihood=poisson_likelihood, inference_method=laplace_inf, kernel=kernelMatern)
        m = GPy.models.GPRegression(X,Y,kernelMatern)
        # fix length scale and variance, leave noise variance unconstrianed
        m.Mat52.variance = self.h**2
        m.Mat52.lengthscale = self.rho
        if holdPara == True:
            m.Mat52.variance.constrain_fixed()
            m.Mat52.lengthscale.constrain_fixed()
        m.optimize()
        return m
    
    def plotGP(self,m,GPwavelInterval=0.002):
        X = m.X
        Y = m.Y
        Xnew = np.arange(X[0],X[-1], step=GPwavelInterval)
        Xnew = Xnew.reshape(len(Xnew),-1)
        mpred = m.predict(Xnew)
        mMean = mpred[0] 
        mCI = m.predict_quantiles(Xnew,quantiles=(2.5, 97.5))         
              
        fig0, ax0 = plt.subplots(figsize=(18,9))
        ax0.plot(X, Y, "bo")
        ax0.set_xlabel("Wavelength [nm]")
        ax0.set_ylabel("Flux [e-/exposure]")
        ax0.grid(True, color='gainsboro', linestyle='-', linewidth=0.5)
        # add posterior predictive mean curve
        ax0.plot(Xnew,mMean ,c="red")
        ax0.fill_between(Xnew.ravel(), mCI[0].ravel(), mCI[1].ravel() , alpha=0.3)
        plt.show()
        
    def PredOneGP(self, suborders, wavelInterval = 0.002):
        count = 0
        timesum = 0
        dataFrame = pd.DataFrame({'wavel':[],'mean':[],'var':[],'separation':[]})
        for i in range(len(suborders)):
            for j in range(len(suborders[i])):
                # timeStart = time.perf_counter()
                count += 1
                m = self.regression(suborders[i][j])
                X = m.X
                Xnew = np.arange(X[0],X[-1], step = wavelInterval)
                wavel = Xnew.reshape(len(Xnew),-1)
                mpred = m.predict(wavel)
                mMean = mpred[0].ravel()
                mVar = mpred[1].ravel()
                separation = np.zeros(len(mMean),dtype=bool)
                # set suborder's starting index as 1
                separation[0] = 1
                tmp = pd.DataFrame({'wavel':wavel.ravel(),'mean':mMean,'var':mVar,'separation':separation})
                dataFrame = pd.concat([dataFrame, tmp], ignore_index=True)
                
                # timeEnd = time.perf_counter()-timeStart
                # timesum += timeEnd
        #         print('{0}:{1}'.format(count,timeEnd))
        # print('total time : {}'.format(timesum))
        return dataFrame
    
    def kernel_Mat52(self, x1, x2):
        tau = np.sqrt(5*(np.subtract.outer(x1,x2)**2))
        kout = self.h**2*(1 + tau/self.rho + tau**2/(3*self.rho**2))*np.exp(-tau/self.rho)
        return kout
    
    # Gradiant calculation: dK/dv
    # input X is the wavelength without considering v
    def kernel_Mat52_grad(self,X,v):   
                  
        def tau_grad2(v,lam1,lam2):
            c = self.c
            term1 = -2*c/(c+v)**2
            term2 = 0.5*((c-v)/(c+v))**(-0.5)
            return np.sqrt(5*np.subtract.outer(lam1 , lam2)**2)*term2*term1      
             
        def tau_grad3(v,lam1,lam2):
            c = self.c
            term1 = np.sqrt((1 - v/c)/(1 + v/c))
            term2 = np.sign(np.subtract.outer(lam1*term1,lam2))
            term3 = -2*c/(c+v)**2
            term4 = 0.5*((c-v)/(c+v))**(-0.5)
            out1 = np.sqrt(5)*lam1*term3*term4
            out2 = np.tile(out1, (len(out1),1))
            return out2*term2
        
        length = len(X)
        matrix1 = tau_grad2(v,X,X)
        matrix2 = tau_grad3(v,X,X)
        matrix3 = -matrix2
        matrix4 = np.zeros([length,length])
        matrixUpper = np.concatenate((matrix1, matrix2),axis=1)
        matrixLower = np.concatenate((matrix3, matrix4),axis=1)
        return np.concatenate((matrixUpper, matrixLower)) 
    
    # Log likelihood with Cholosky decompostion    
    def logLik_Chol(self,wavel1, wavel2, Y12, var1, var2, v, evalGradiant = False):
        # startTime = time.clock()
        length = len(wavel1)
        wavel1 = wavel2/np.sqrt((1 + v/self.c)/(1 - v/self.c))
        X12 = np.concatenate((wavel1,wavel2))
        
        # varMatrix = np.diag(np.concatenate((np.ones(length)*var1,np.ones(length)*var2)))
        varMatrix = np.diag(np.concatenate((var1,var2)))
        K12 = self.kernel_Mat52(X12,X12)
        K12 += varMatrix
        # scipy package solution
        Y12  = Y12[:, np.newaxis]
        L = scipy.linalg.cholesky(K12 + (1e-10*np.identity(2*length)),lower=True)
        alpha = scipy.linalg.cho_solve((L,True),Y12)
        loglik = -0.5 * np.einsum("ik,ik->k", Y12, alpha)
        loglik -= np.log(np.diag(L)).sum()
        loglik -= length * np.log(2 * np.pi)     
        # # numpy package solution
        # L = np.linalg.cholesky(K12 + (1e-10*np.identity(2*length)))
        # alpha = np.linalg.solve(L.T,np.linalg.solve(L,Y12))        
        # loglik = -0.5*(Y12.T @ alpha) - np.log(L.diagonal()).sum() - length*np.log(2*np.pi)
        
        negLogLik = - loglik
        # print((time.clock() - startTime))
        
        # Evaluate gradient
        if evalGradiant == True:
            K_gradient = self.kernel_Mat52_grad(wavel2,v)
            tmp = np.einsum("ik,jk->ijk", alpha, alpha)  # k: output-dimension
            tmp -= scipy.linalg.cho_solve((L, True), np.eye(2*length))[:, :, np.newaxis]
            # tmp -= np.linalg.solve(L.T,np.linalg.solve(L,np.eye(2*length)))[:, :, np.newaxis]
            # Compute "0.5 * trace(tmp.dot(K_gradient))" without
            # constructing the full matrix tmp.dot(K_gradient) since only
            # its diagonal is required
            log_likelihood_gradient = 0.5 * np.einsum("ijl,jik->kl", tmp, K_gradient[:,:,np.newaxis])        
            return negLogLik.item(), log_likelihood_gradient.sum(-1)
        else:
            return negLogLik.item()
        
    def optimizeRV(self, GPFitted1, GPFitted2, loglik_nearMax = False, deltaV = 1):
        len1 = len(GPFitted1)
        len2 = len(GPFitted2)
        if len1 != len2:
            assert abs(len1-len2)<5
            print("WARNING: Unequal suborder length, removing the difference")
            if len1 > len2:
                GPFitted1 = GPFitted1.iloc[:len2]
            else:
                GPFitted2 = GPFitted2.iloc[:len1]
        Y12 = np.concatenate((GPFitted1['mean'].to_numpy(),GPFitted2['mean'].to_numpy()))
        # startTime = time.clock()               
        objFun = lambda v:self.logLik_Chol(GPFitted1['wavel'].to_numpy(), GPFitted2['wavel'].to_numpy(), 
                                           Y12, GPFitted1['var'].to_numpy(), GPFitted2['var'].to_numpy(), v)
        # step_count = opt.minimize(objFun, [v]).numpy()
        # result = v.numpy()
        result = minimize_scalar(objFun,method='brent',tol=1e-3)
        # result = scipy.optimize.minimize(objFun, x0=-20, method="L-BFGS-B", options={'ftol': 1e-15, 'gtol': 1e-15, 'eps': 1e-15})
        # result = scipy.optimize.minimize(objFun, 500, method="Powell", options={'disp': True,'ftol':1e-5,'xtol': 1e-5})
        # result = optimize.dual_annealing(objFun,bounds = list(zip([-10000], [10000])))
        
        ### Following eq.11 in Zechmeister1,2017: error estimation under porabolic approximation
        if loglik_nearMax == True:
            maxmum = result.x
            neglog_max = result.fun
            neglog_maxN = objFun(maxmum - deltaV)
            neglog_maxP = objFun(maxmum + deltaV)
            uncertainty = 2*deltaV**2/(neglog_maxN - 2*neglog_max + neglog_maxP)
            return  maxmum,uncertainty
        else:
            return result.x.item()
        