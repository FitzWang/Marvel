# %% Imports

import os
import numpy as np
import pandas as pd
import scipy as sp

from numpy import exp

from scipy.integrate import trapz
from scipy.interpolate import griddata, CubicSpline, Akima1DInterpolator
from scipy.signal import gaussian

from numba import jit, int64, float64

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator

from numba import jit




# %% Matplotlib configuration

plt.rc('font',   size=14)          # controls default text sizes
plt.rc('axes',   titlesize=15)     # fontsize of the axes title
plt.rc('axes',   labelsize=15)     # fontsize of the x and y labels
plt.rc('xtick',  labelsize=15)     # fontsize of the tick labels
plt.rc('ytick',  labelsize=15)     # fontsize of the tick labels
plt.rc('legend', fontsize=14)       # legend fontsize
plt.rc('figure', titlesize=14)     # fontsize of the figure title

matplotlib.rcParams['text.usetex'] = False



# %% Auxilliary functions and constants

def sec(angle):
    return 1.0/np.cos(angle)

PI = np.pi
C = 299792458.0                # Speed of light     [m/s]
H = 6.6260755e-27              # Constant of Planck [erg s]




# %% A Rebin function

def rebin(a, binSize, func):
    """
    Rebin a 1D array: partition an array 'a' into consecutive chunks of size 'binSize',
    and apply the function 'func' to each of these chunks. Discard the edge with a
    left-over smaller chunk.

    INPUT:
        a:       a 1D numpy array
        binsize: size of the bins (integer)
        func:    function to be applied to each bin (e.g. np.sum)

    OUTPUT:
        b : a rebinned 1D array

    EXAMPLE:
        >>> x = np.arange(13)
        >>> print(x)
        [ 0  1  2  3  4  5  6  7  8  9 10 11 12]
        >>> y = rebin(x, 3, np.sum)
        >>> print(y)
        [ 3 12 21 30]
    """

    sizeRebinned = int(np.floor(len(a)/binSize))
    b = np.array([func(a[n*binSize:(n+1)*binSize]) for n in range(sizeRebinned)])
    return b





@jit((float64[:], float64[:], float64[:], float64[:], float64), nopython=True)
def convolve(wavel, linespec, contspec, newwavel, R):
    """
    Convolve the given spectrum (wavel, spec) with a gaussian with a stdev determined by resolving power R
    and evaluate the result in the 'newwavel' points. This function differs from the numpy convolve() function
    as it allows non-equidistant wavelength points. It computes the convolution with a direct sum.

    INPUT:
        wavel:    wavelength points of the unconvolved spectrum, not necessarily equidistant, but assumed to be sorted
        spec:     spectrum, not necessarily normalized
        newwavel: wavelength points of the convolved spectrum, assumed to be sorted
        R:        Resolving power of the spectrograph

    OUTPUT:
        newspec:  convolved spectrum, same size as newwavel
    """

    newlinespec = np.zeros_like(newwavel)
    newcontspec = np.zeros_like(newwavel)


    k0 = 0
    for n in range(len(newwavel)):
        sumWeights = 0.0
        sigma = newwavel[n] / R / 2.355         # 2.355 to convert from FWHM to sigma
        factor = 0.5/sigma**2
        for k in range(k0, len(wavel)):
            delta = wavel[k] - newwavel[n]
            if delta < -6 * sigma: 
                # Move the left most point of the 6-sigma interval along, so that we don't have to
                # go through the wavelength array from the very beginning each time. However, build
                # in a safety margin of 10 points, because the 6-sigma interval is enlarging with
                # larger wavelengths.
                k0 = max(0, k-10)
                continue
            if delta > 6 * sigma: break
            weight = exp(-factor*delta**2) 
            newlinespec[n] += weight * linespec[k]
            newcontspec[n] += weight * contspec[k]
            sumWeights += weight
        newlinespec[n] /= sumWeights
        newcontspec[n] /= sumWeights

    return newlinespec, newcontspec






# %% A function to load a spectrum, convolve and rebin it

def loadSpectrum(Teff, logg, vmicro, FeH, vsini, R=None, lambdaEff=600., samplingResolution=3, velocityShift=0.0):
    """
    INPUT:
        Teff:               [K]
        logg:               [cgs]
        vmicro:             micro-turbulent velocity [km/s]
        FeH:                [Fe/H]
        vsini:              [km/s]
        R:                  Resolving power
        lambdaEff:          effective wavelength of the spectrograph's throughput function [nm]
        samplingResolution: number of wavelength points per FWHM resolution element (real-valued)
        velocityShift:      Doppler shift the entire spectrum with the given velocity [m/s]

    OUTPUT:
        data: pandas DataFrame
              ['wavel']    : wavelength     [nm]
              ['lineFlux'] : line flux      [erg/s/cm^2/nm]
              ['contFlux'] : continuum flux [erg/s/cm^2/nm]
    """

    path = os.getcwd() + "/../data/lp0000_{0:05d}_{1:04d}_{2:04d}_{3:04d}_Vsini_{4:04d}.rgs.gz".format(Teff, int(round(100*round(logg,2))), int(vmicro*10), int(FeH), int(vsini))
    if not os.path.isfile(path):
        print("Error: filename: " + path + " does not exist: skipping")
        return None
    else:
        print("Loading file " + path)

    data = pd.read_csv(path, usecols=[0,2,3], names=['wavel', 'lineFlux', 'contFlux'], delim_whitespace=True)
    data['wavel'] /= 10.0                                   # Angstrom -> nm
    data['lineFlux'] *= C * 1.e9 / data['wavel']**2         # F_nu [erg/s/cm^2/Hz] -> F_lambda  [erg/s/cm^2/nm]
    data['contFlux'] *= C * 1.e9 / data['wavel']**2         # F_nu [erg/s/cm^2/Hz] -> F_lambda  [erg/s/cm^2/nm]


    if velocityShift != 0.0:

        # Doppler-shift the entire spectrum

        data['wavel'] += velocityShift / 299792458.0 * data['wavel']

        # Resample the spectrum again to an equidistant wavelength grid (needed to convolve to a specific R)

        newWavel = np.linspace(data['wavel'].min(), data['wavel'].max(), len(data))
        lineFluxInterpolator = CubicSpline(data['wavel'], data['lineFlux'])
        contFluxInterpolator = CubicSpline(data['wavel'], data['contFlux'])
        newLineFlux = lineFluxInterpolator(newWavel)
        newContFlux = contFluxInterpolator(newWavel)

        # Replace the old data frame

        data = pd.DataFrame.from_dict({'wavel': newWavel, 'lineFlux': newLineFlux, 'contFlux': newContFlux})


    if R is not None:
        # Convolve to the required resolving power R

        wavelStep = data.loc[1,'wavel'] - data.loc[0,'wavel']
        FWHM = lambdaEff / R                                         # FWHM [nm]
        sigma = FWHM / 2.355 / wavelStep                             # For a Gaussian: FWHM = 2.355 * sigma [array elements]
        kernel = sp.signal.gaussian(100, std=sigma)
        kernel /= kernel.sum()
        lineFlux = np.convolve(data['lineFlux'], kernel, "same")
        contFlux = np.convolve(data['contFlux'], kernel, "same")

        # Resample to the required sampling resolution

        lineFluxInterpolator = CubicSpline(data['wavel'], lineFlux)               # [erg/s/cm^2/nm]
        contFluxInterpolator = CubicSpline(data['wavel'], contFlux)               # [erg/s/cm^2/nm]
        newDeltaLambda = FWHM / samplingResolution                                # [nm]
        newWavel = np.arange(data['wavel'].min(), data['wavel'].max(), newDeltaLambda)
        newLineFlux = lineFluxInterpolator(newWavel)
        newContFlux = contFluxInterpolator(newWavel)

        # Replace the old data frame

        data = pd.DataFrame.from_dict({'wavel': newWavel, 'lineFlux': newLineFlux, 'contFlux': newContFlux})

    return data















# %% An alternative function to load a spectrum, convolve and rebin it
#    It differs from loadspectrum() in the sense that it does not use any interpolation.

def loadSpectrum2(Teff, logg, vmicro, FeH, vsini, R=None, lambdaEff=600., samplingResolution=3, velocityShift=0.0):
    """
    INPUT:
        Teff:               [K]
        logg:               [cgs]
        vmicro:             micro-turbulent velocity [km/s]
        FeH:                [Fe/H]
        vsini:              [km/s]
        R:                  Resolving power
        lambdaEff:          effective wavelength of the spectrograph's throughput function [nm]
        samplingResolution: number of wavelength points per FWHM resolution element (real-valued)
        velocityShift:      Doppler shift the entire spectrum with the given velocity [m/s]

    OUTPUT:
        data: pandas DataFrame
              ['wavel']    : wavelength     [nm]
              ['lineFlux'] : line flux      [erg/s/cm^2/nm]
              ['contFlux'] : continuum flux [erg/s/cm^2/nm]
    """

    path = os.getcwd() + "/../data/lp0000_{0:05d}_{1:04d}_{2:04d}_{3:04d}_Vsini_{4:04d}.rgs.gz".format(Teff, int(round(100*round(logg,2))), int(vmicro*10), int(FeH), int(vsini))
    if not os.path.isfile(path):
        print("Error: filename: " + path + " does not exist: skipping")
        return None
    else:
        print("Loading file " + path)

    data = pd.read_csv(path, usecols=[0,2,3], names=['wavel', 'lineFlux', 'contFlux'], delim_whitespace=True)
    data['wavel'] /= 10.0                                   # Angstrom -> nm
    data['lineFlux'] *= C * 1.e9 / data['wavel']**2         # F_nu [erg/s/cm^2/Hz] -> F_lambda  [erg/s/cm^2/nm]
    data['contFlux'] *= C * 1.e9 / data['wavel']**2         # F_nu [erg/s/cm^2/Hz] -> F_lambda  [erg/s/cm^2/nm]

    minWavelength = data['wavel'].min()
    maxWavelength = data['wavel'].max()

    if velocityShift != 0.0:

        # Doppler-shift the entire spectrum

        data['wavel'] += velocityShift / 299792458.0 * data['wavel']


    if R is not None:
        # Convolve to the required resolving power R
        # TODO: don't take a fixed FWHM, but one that varies over the spectrum: lambda/R

        wavelStep = data.loc[1,'wavel'] - data.loc[0,'wavel']
        FWHM = lambdaEff / R                                                      # FWHM [nm]
        newDeltaLambda = FWHM / samplingResolution                                # [nm]
        newWavel = np.arange(minWavelength, maxWavelength, newDeltaLambda)

        lineFlux, contFlux = convolve(data['wavel'].values, data['lineFlux'].values, data['contFlux'].values, newWavel, R)

        # Replace the old data frame

        data = pd.DataFrame.from_dict({'wavel': newWavel, 'lineFlux': lineFlux, 'contFlux': contFlux})

    return data















# %% A function to compute the electron flux on Earth

def electronFlux(spectrum, telescopeArea, throughputInterpolator, lambdaMin, lambdaMax, exposureTime, Vmag, includePoissonNoise=False, normalizeSpectrum=False):

    """
    INPUT:
        spectrum:               Output of loadSpectrum()
        telescopeArea:          [cm^2]
        throughputInterpolator: Interpolating function of the throughput
        lambdaMin:              Lower wavelength limit where the interpolator is applicable   [nm]
        lambdaMax:              Higher wavelength limit where the interpolator is applicable  [nm]
        exposureTime:           [s]
        Vmag:                   magnitude
        includePoissonNoise:    True or False
        normalizeSpectrum:      True or False, normalize out the throughput and the continuum flux. Keep the noise.
    OUTPUT:
        data:
            ['NelectronsLine'] : Line electron flux        [electrons/exposure/pixel]
            ['NelectronsCont'] : Continuum electron flux   [electrons/exposure/pixel]
    """

    # Make a deep copy of the spectrum so that the original dataframe is untouched

    data = spectrum.copy(deep=True)

    # Rescale the spectrum according to the star's magnitude

    lambda_eff_V = 545.0                                               # Effective wavelength of the Johnson V band  [nm]
    F0_V = 3.631e-8                                                    # Zeropoint of the Johnson V band  [erg/s/cm^2/nm]
    contFluxInterpolator = CubicSpline(data['wavel'], data['contFlux'])                                 # [erg/s/cm^2/nm]
    fluxAtStellarSurfaceAtEffWavel = contFluxInterpolator(lambda_eff_V)                                 # [erg/s/cm^2/nm]
    fluxAtEarthSurfaceAtEffWavel = F0_V * 100**(-Vmag/5)
    lineFluxVmag = data['lineFlux'] / fluxAtStellarSurfaceAtEffWavel * fluxAtEarthSurfaceAtEffWavel     # [erg/s/cm^2/nm]
    contFluxVmag = data['contFlux'] / fluxAtStellarSurfaceAtEffWavel * fluxAtEarthSurfaceAtEffWavel     # [erg/s/cm^2/nm]

    # Convert from energy flux to photon flux

    deltaLambda = data['wavel'].diff()                                                                                     # [nm]
    deltaLambda.iloc[0] = deltaLambda.iloc[1]               # Avoid NaN for 1st element because of backward difference
    energyOnePhoton = H * C * 1e9 / data['wavel']                                                                          # [erg]
    NphotonsLine = lineFluxVmag / energyOnePhoton * deltaLambda * telescopeArea * exposureTime    # [photons/exposure]
    NphotonsCont = contFluxVmag / energyOnePhoton * deltaLambda * telescopeArea * exposureTime    # [photons/exposure]

    # Compute the log-derivative which will be used for RV precision computations

    logPhotonFlux = CubicSpline(np.log(data['wavel'].values), np.log(NphotonsLine))
    logDerivative = logPhotonFlux.derivative(1)
    data['logDerivative'] = logDerivative(np.log(spectrum['wavel'].values))

    # Take the spectrograph's throughput curve into account (this includes the QE)

    throughput = throughputInterpolator(data['wavel'])
    throughput[np.isnan(throughput)] = 1.e-6
    throughput[throughput == 0.0] = 1.e-6
    data['NelectronsLine'] = NphotonsLine * throughput                        # [photons/exposure]
    data['NelectronsCont'] = NphotonsCont * throughput                        # [photons/exposure]


    # Cut everything outside the interpolator's applicability range
    # The deep copy is to avoid the "assignment to a copy" warning, when you afterwards want to add columns
    # to the dataframe that we return

    insideRange = (data['wavel'] > lambdaMin) & (data['wavel'] < lambdaMax)
    result = data[insideRange].copy(deep=True)
    result.reset_index(drop=True, inplace=True)

    # If asked for include Poisson noise, include de noise in the line flux. Not in the continuum flux
    # so that we can normalize the spectrum afterwards.

    if includePoissonNoise:
        result['NelectronsLine'] = np.random.poisson(lam=np.array(result['NelectronsLine']))


    # If asked, normalize the spectrum.

    if normalizeSpectrum:
        result['NelectronsLine'] /= result['NelectronsCont']

    return result



















# %% A function to compute a normalized spectrum with a fixed (wavelength-independent) S/N ratio

def normalizedSpectrumxWithFixedSN(spectrum, lambdaMin, lambdaMax, SNratio):

    """
    INPUT:
        spectrum:               Output of loadSpectrum()
        lambdaMin:              Lower wavelength limit where the interpolator is applicable   [nm]
        lambdaMax:              Higher wavelength limit where the interpolator is applicable  [nm]
        SNratio:                S/N ratio of the spectrum (e.g. 200)
    OUTPUT:
        data:
            ['wavel']          : Wavelength of the spectrum
            ['NelectronsLine'] : Normalized line electron flux
    """

    # Make a deep copy of the spectrum so that the original dataframe is untouched

    data = spectrum.copy(deep=True)

    # Convert from energy flux to photon flux

    deltaLambda = data['wavel'].diff()                                                                                     # [nm]
    deltaLambda.iloc[0] = deltaLambda.iloc[1]               # Avoid NaN for 1st element because of backward difference
    energyOnePhoton = H * C * 1e9 / data['wavel']                                                                          # [erg]
    data['NelectronsLine'] = data['lineFlux'] / energyOnePhoton * deltaLambda 
    data['NelectronsCont'] = data['contFlux'] / energyOnePhoton * deltaLambda

    # Cut everything outside the interpolator's applicability range
    # The deep copy is to avoid the "assignment to a copy" warning, when you afterwards want to add columns
    # to the dataframe that we return

    insideRange = (data['wavel'] > lambdaMin) & (data['wavel'] < lambdaMax)
    result = data[insideRange].copy(deep=True)
    result.reset_index(drop=True, inplace=True)

    # If asked for include Poisson noise, include de noise in the line flux. Not in the continuum flux
    # so that we can normalize the spectrum afterwards.

    result['NelectronsLine'] = np.random.normal(result['NelectronsLine'].values, result['NelectronsLine'].values / SNratio)


    # Normalize the spectrum.

    result['NelectronsLine'] /= result['NelectronsCont']

    return result


















# %% Function to compute the Vrad precision and the quality of the spectrum

def vradPrecision(spectrum, Ntelescopes=1, telluricMask=None, sigmaRON=0, Npix=0):

    """
    INPUT:
        spectrum:     output of electronFlux(), the stellar spectrum of 1 telescope
        telluricMask: array: 0 if there is a telluric line at the wavelength, 1 if there is not.
        sigmaRON:     readout noise [e-/pixel]
        Npix:         number of pixels in the cross-dispersion direction
    """

    if telluricMask is None:
        mask = np.ones(len(spectrum), dtype=np.bool)
    else:
        mask = np.array(telluricMask, dtype=np.bool)

    spectrum['varVel'] = C**2 * (spectrum['NelectronsLine'] + Npix * sigmaRON**2)                             \
                              / spectrum['NelectronsLine']**2 / spectrum['logDerivative']**2 / Ntelescopes
    totalVelocityPrecision = np.sqrt(1./ np.sum(1.0/spectrum[mask]['varVel']))
    Q = np.sqrt(np.sum(spectrum[mask]['NelectronsLine'] * spectrum[mask]['logDerivative']**2) / np.sum(spectrum[mask]['NelectronsLine']))

    return spectrum, totalVelocityPrecision, Q








# %% Spectrograph configuration

def getSpectrograph(name):
    """
    INPUT:
        name: either "Coralie" or "Marvel"

    OUTPUT:
        R:                      resolving power
        telescopeArea:          [cm^2]
        effWavelSpectrograph:   effective wavelength of the spectrograph [nm]
        samplingResolution:     Nr of wavelength points per FWHM bin
        throughputInterpolator: Interpolating function of the throughput (QE included)
        lambdaMin:              Lower wavelength limit where the interpolator is applicable   [nm]
        lambdaMax:              Higher wavelength limit where the interpolator is applicable  [nm]
    """

    if name == "Marvel":       # Marvel or Coralie
        R = 92000.
        telescopeArea = PI * (80.0/2)**2      # Throughput function takes into account the effect of the central obstruction of 20 cm
        effWavelSpectrograph = 620.0
        samplingResolution = 3.0              # [pix/FWHM_lambda_resolution_element]
        table = pd.read_csv("MarvelIntraOrderThroughput.txt", sep='\s+', names=['wavel', 'throughput'])
        throughputInterpolator = Akima1DInterpolator(table['wavel'], table['throughput'])
        lambdaMin = table['wavel'].min()
        lambdaMax = table['wavel'].max()
    elif name == "MarvelNoOrders":
        R = 92000.
        telescopeArea = PI * (80.0/2)**2      # Throughput function takes into account the effect of the central obstruction of 20 cm
        effWavelSpectrograph = 620.0
        samplingResolution = 3.0                                     # [pix/FWHM_lambda_resolution_element]
        table = pd.read_csv("marvelThroughputCurve.txt", sep='\s+', names=['wavel', 'throughput'])
        throughputInterpolator = Akima1DInterpolator(table['wavel'], table['throughput'])
        lambdaMin = table['wavel'].min()
        lambdaMax = table['wavel'].max()
    elif name == "Coralie":
        R = 85000.
        telescopeArea = PI * ((120.0/2)**2 - (30.0/2)**2)
        effWavelSpectrograph = 558.0
        samplingResolution = 5.0
        table = pd.read_csv("CoralieEfficiency.txt", sep='\s+', names=['wavel', 'throughput'])
        throughputInterpolator = Akima1DInterpolator(table['wavel'], table['throughput'])
        lambdaMin = table['wavel'].min()
        lambdaMax = table['wavel'].max()
    else:
        print("Only 'Marvel', 'MarvelNoOrders' or 'Coralie' allowed as input name")

    return R, telescopeArea, effWavelSpectrograph, samplingResolution, throughputInterpolator, lambdaMin, lambdaMax
