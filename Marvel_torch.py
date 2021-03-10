# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 10:54:23 2021

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
import torch
import gpytorch

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
        self.initial_noiseVar = 0.0005
    
    def regression(self,spectrum, graph = False, cuda = False):
        class ExactGPModel(gpytorch.models.ExactGP):
            def __init__(self, train_x, train_y, likelihood):
                super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
                self.mean_module = gpytorch.means.ConstantMean()
                self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu = 2.5))
        
            def forward(self, x):
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)    
        
        # define data
        wavelStart = spectrum["wavel"].iloc[0]
        wavelEnd = spectrum["wavel"].iloc[-1]
        X = torch.from_numpy(spectrum["wavel"].ravel())
        Y = torch.from_numpy(spectrum["NelectronsLine"].ravel())
        # Define parameters
        hypers = {
            'likelihood.noise_covar.noise': torch.tensor(self.initial_noiseVar),
            'covar_module.base_kernel.lengthscale': torch.tensor(self.rho),
            'covar_module.outputscale': torch.tensor(self.h),
        }
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPModel(X, Y, likelihood).double()
        model.initialize(**hypers)
        if cuda == True:
            assert torch.cuda.is_available() == True,"Cuda is nor available, Please install pytorch_gpu"
            X = X.cuda()
            Y = Y.cuda()
            model = model.cuda()
            likelihood = likelihood.cuda()
        smoke_test = ('CI' in os.environ)
        try:
            smoke_test == True
        except OSError:
            print("smoke test failed")            
        training_iter = 40
        # Find optimal model hyperparameters
        model.train()
        likelihood.train()        
        # Use the adam optimizer
        optimizer = torch.optim.Adam([{"params": model.likelihood.parameters()},], lr=0.1)
        # optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters
        
        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        # Optimization Procedure
        for i in range(training_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = model(X)
            # Calc loss and backprop gradients
            loss = -mll(output, Y)
            loss.backward()
            # print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.5f' % (
            #     i + 1, training_iter, loss.item(),
            #     model.covar_module.base_kernel.lengthscale.item(),
            #     model.likelihood.noise.item()
            # ))
            optimizer.step()
        if graph == True:
            # Get into evaluation (predictive posterior) mode
            model.eval()
            likelihood.eval()
            if cuda == True:
                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    observed_pred = likelihood(model(X))
                    mean = observed_pred.mean
                    lower, upper = observed_pred.confidence_region()                    
                mean = mean.cpu()
                lower = lower.cpu()
                upper = upper.cpu()
                
                train_x = X.cpu()
                train_y = Y.cpu()
                test_x = X.cpu()
                with torch.no_grad():
                    # Initialize plot
                    f, ax = plt.subplots(1, 1, figsize=(16, 9))                
                    # Plot training data as black stars
                    ax.plot(train_x.numpy(), train_y.numpy(), 'k*')
                    # Plot predictive means as blue line
                    ax.plot(test_x.numpy(), mean.numpy(), 'b')
                    # Shade between the lower and upper confidence bounds
                    ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
                    ax.set_ylim([-0.1, 1.3])
                    ax.legend(['Observed Data', 'Mean', 'Confidence'])
            # Cuda is not used
            else:            
                # Test points are regularly spaced along [0,1]
                # Make predictions by feeding model through likelihood
                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    test_x = X
                    observed_pred = likelihood(model(test_x))               
                with torch.no_grad():
                    # Initialize plot
                    f, ax = plt.subplots(1, 1, figsize=(16, 9))           
                    # Get upper and lower confidence bounds
                    lower, upper = observed_pred.confidence_region()
                    # Plot training data as black stars
                    ax.plot(X.numpy(), Y.numpy(), 'k*')
                    # Plot predictive means as blue line
                    ax.plot(test_x.numpy(), observed_pred.mean.numpy(), 'b')
                    # Shade between the lower and upper confidence bounds
                    ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
                    ax.set_ylim([-0.1, 1.3])
                    ax.legend(['Observed Data', 'Mean', 'Confidence'])
        if cuda == True:
            model.eval()
            likelihood.eval()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                test_x = np.arange(wavelStart,wavelEnd, step=self.GPwavelInterval)
                observed_pred = likelihood(model(torch.from_numpy(test_x).cuda()))
                mean = observed_pred.mean.cpu().detach().numpy()
                variance = observed_pred.variance.cpu().detach().numpy()
            return test_x, mean, variance
        else:
            model.eval()
            likelihood.eval()           
            # Test points are regularly spaced along [0,1]
            # Make predictions by feeding model through likelihood
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                test_x = np.arange(wavelStart,wavelEnd, step=self.GPwavelInterval)
                observed_pred = likelihood(model(torch.from_numpy(test_x)))
                mean = observed_pred.mean.cpu().detach().numpy()
                variance = observed_pred.variance.cpu().detach().numpy()
            return test_x, mean, variance
    
    # def plotGP(self,m):
    #     X = m.X
    #     Y = m.Y
    #     Xnew = np.arange(X[0],X[-1], step=self.GPwavelInterval)
    #     Xnew = Xnew.reshape(len(Xnew),-1)
    #     mpred = m.predict(Xnew)
    #     mMean = mpred[0]
    #     mVar = mpred[1]    
    #     mCI = m.predict_quantiles(Xnew,quantiles=(2.5, 97.5))         
              
    #     fig0, ax0 = plt.subplots(figsize=(18,9))
    #     ax0.plot(X, Y, "bo")
    #     ax0.set_xlabel("Wavelength [nm]")
    #     ax0.set_ylabel("Flux [e-/exposure]")
    #     ax0.grid(True, color='gainsboro', linestyle='-', linewidth=0.5)
    #     # add posterior predictive mean curve
    #     ax0.plot(Xnew,mMean ,c="red")
    #     ax0.fill_between(Xnew.ravel(), mCI[0].ravel(), mCI[1].ravel() , alpha=0.3)
    #     plt.show()
                
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
        
    # Log likelihood with Cholosky decompostion    
    def logLik_Chol(self,wavel1, wavel2, Y12, var1, var2, v, evalGradiant = False):
        # startTime = time.clock() 
        length = len(wavel1)
        wavel1 = wavel1/np.sqrt((1 + v/self.c)/(1 - v/self.c))
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
    
    def optimizeRV(self, spectrum1, spectrum2,cuda = False , loglik_nearMax = False):
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
        Xnew1,mMean1,mVar1 = self.regression(spectrum1,cuda = cuda)
        Xnew2,mMean2,mVar2  = self.regression(spectrum2,cuda = cuda)

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
