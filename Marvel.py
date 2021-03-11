# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 10:54:23 2021

@author: guang
"""
import GPy
import numpy as np

import time
from utilities import *
import matplotlib.pyplot as plt
import pandas as pd
import scipy
from scipy import stats, signal
from scipy.optimize import minimize_scalar
from scipy import optimize
from scipy.spatial.distance import pdist, cdist, squareform


class Spectrum:
    def __init__(self,velocityShift = None, path = None, nPointSuborder = 450, Vmag = 9., exposureTime = 900.):
        if velocityShift == None:
            assert path != None, 'Please provide velocity shilf value OR path of spectrum'
            self.spectrum = pd.read_csv(path)
        else:
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
            self.spectrum = spectrum
            
        self.length = len(self.spectrum)
        # Split spectrum to orders via trough of original signal
        miniInd = signal.argrelextrema(self.spectrum['NelectronsCont'].ravel(), np.less, order = 10)
        # remove the first split threshold point, since first order only contains few points
        miniInd = miniInd[0].tolist()[1:]
        self.spectrumByOrders = np.split(self.spectrum, miniInd, axis=0)
        self.nSplit = [round(len(i)/nPointSuborder) for i in self.spectrumByOrders]
        
    
    def orderToSuborder(self,spectrum,nSplit):
        '''
        To separate spectrum to n sub-spectra

        '''
        pointsPerOrder = int(np.ceil(len(spectrum)/nSplit))
        suborder = []
        for i in range(nSplit-1):
            suborder.append(spectrum.iloc[pointsPerOrder*i:pointsPerOrder*(1+i)])
        # for last sub-order, select obs to the end in case obs lost because of ceiling
        i = i+1
        suborder.append(spectrum.iloc[pointsPerOrder*i:])
        return suborder
    
    def spectrumToSuborder(self,nSplit = None):
        if nSplit == None:
            nSplit = self.nSplit
        assert len(self.spectrumByOrders) == len(nSplit)
        suborders = []
        for i in range(len(nSplit)):
            if nSplit[i] == 0:
                suborders.append([self.spectrumByOrders[i]])
            else:
                suborders.append(self.orderToSuborder(self.spectrumByOrders[i],nSplit[i]))
        return suborders    
        
    
    def subSpectrum(self, lowerWavl: float, upperWavl:float):
        '''
        To select spectrum with specific range of wavelength
        
        '''
        subSpec = self.spectrum.loc[(self.spectrum["wavel"] > lowerWavl) & (self.spectrum["wavel"] < upperWavl)]
        noPoints = len(subSpec)
        return subSpec, noPoints
    
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
        

class GPR:

    def __init__(self,h = 0.5,rho = 0.02):
        self.h = h
        self.rho = rho
        self.c = 299792458
        # Global parameter definition (nm)
        self.GPwavelInterval = 0.002
    
    def regression(self,spectrum):
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
        m.Mat52.variance.constrain_fixed()
        m.Mat52.lengthscale.constrain_fixed()
        m.optimize()        
        # print(m)
        return m                                 
    
    def plotGP(self,m):
        X = m.X
        Y = m.Y
        Xnew = np.arange(X[0],X[-1], step=self.GPwavelInterval)
        Xnew = Xnew.reshape(len(Xnew),-1)
        mpred = m.predict(Xnew)
        mMean = mpred[0]
        mVar = mpred[1]    
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
        
        
    def kernel_Mat52(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
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
    
    # compute only K12 matrix for joint 2 spectra
    def Kmatrix(self,spectrum1, spectrum2, v,imshow = True):
        len1 = len(spectrum1)
        len2 = len(spectrum2)
        if len1 != len2:
            assert abs(len1-len2)<5
            print("WARNING: Unequal suborder length, removing the difference")
            if len1 > len2:
                spectrum1 = spectrum1.iloc[:len2]
            else:
                spectrum2 = spectrum2.iloc[:len1]
        m1 = self.regression(spectrum1)
        m2 = self.regression(spectrum2)
        X1 = m1.X
        Xnew1 = np.arange(X1[0],X1[-1], step=self.GPwavelInterval)
        wavel1 = Xnew1.reshape(len(Xnew1),-1)
        mpred1 = m1.predict(wavel1)
        mVar1 = mpred1[1].ravel()
        
        X2 = m2.X
        Xnew2 = np.arange(X2[0],X2[-1], step=self.GPwavelInterval)
        wavel2 = Xnew2.reshape(len(Xnew2),-1)
        mpred2 = m2.predict(wavel2)
        mVar2 = mpred2[1].ravel()
        
        Xnew1 = Xnew2/np.sqrt((1 + v/self.c)/(1 - v/self.c))
        X12 = np.concatenate((Xnew1,Xnew2))
        
        # varMatrix = np.diag(np.concatenate((np.ones(length)*var1,np.ones(length)*var2)))
        varMatrix = np.diag(np.concatenate((mVar1,mVar2)))
        K12 = self.kernel_Mat52(X12,X12)
        K12 += varMatrix
        if imshow == True:
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(K12, cmap = 'gray')
        return K12
    
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
            return negLogLik, log_likelihood_gradient.sum(-1)
        else:
            return negLogLik   
    
    def optimizeRV(self, spectrum1, spectrum2, cuda=False, loglik_nearMax = False):
        len1 = len(spectrum1)
        len2 = len(spectrum2)
        if len1 != len2:
            assert abs(len1-len2)<5
            print("WARNING: Unequal suborder length, removing the difference")
            if len1 > len2:
                spectrum1 = spectrum1.iloc[:len2]
            else:
                spectrum2 = spectrum2.iloc[:len1]
        # startTime = time.clock()
        m1 = self.regression(spectrum1)
        m2 = self.regression(spectrum2)
        X1 = m1.X
        Xnew1 = np.arange(X1[0],X1[-1], step=self.GPwavelInterval)
        wavel1 = Xnew1.reshape(len(Xnew1),-1)
        mpred1 = m1.predict(wavel1)
        mMean1 = mpred1[0].ravel()
        mVar1 = mpred1[1].ravel()
        
        X2 = m2.X
        Xnew2 = np.arange(X2[0],X2[-1], step=self.GPwavelInterval)
        wavel2 = Xnew2.reshape(len(Xnew2),-1)
        mpred2 = m2.predict(wavel2)
        mMean2 = mpred2[0].ravel()
        mVar2 = mpred2[1].ravel()
        
        # var1 = self.regression(spectrum1).Gaussian_noise.variance[0]
        # var2 = self.regression(spectrum2).Gaussian_noise.variance[0]
        # print((time.clock() - startTime))
        # wavel1 = spectrum1["wavel"].ravel()
        # wavel2 = spectrum2["wavel"].ravel()
        # Y12 = np.concatenate((spectrum1["NelectronsLine"].ravel(),spectrum2["NelectronsLine"].ravel()))
        Y12 = np.concatenate((mMean1,mMean2))
               
        objFun = lambda v:self.logLik_Chol(Xnew1, Xnew2, Y12, mVar1, mVar2, v)
        # step_count = opt.minimize(objFun, [v]).numpy()
        # result = v.numpy()
        result = minimize_scalar(objFun,method='brent')
        # result = scipy.optimize.minimize(objFun, x0=-50, method="L-BFGS-B", jac=True, options={'disp': True})
        # result = minimize(objFun, 0, method="L-BFGS-B", options={'disp': True})
        # result = optimize.dual_annealing(objFun,bounds = list(zip([-10000], [10000])))
        if loglik_nearMax == True:
            maxmum = result.x.item()
            rvlist = np.arange(-6,7)*5 + maxmum
            # rvlist = np.append(rvlist,-55)
            logliklist = []
            for i in range(len(rvlist)):
                logliklist.append(objFun(rvlist[i]))
            return  maxmum,logliklist
        else:
            return result
        
    
