
from utilities import *
import matplotlib.pyplot as plt

# Load the instrumental parameters for MARVEL

R, telescopeArea, effWavelSpectrograph, samplingResolution, throughputInterpolator, lambdaMin, lambdaMax = getSpectrograph("Marvel")

# Choose the stellar parameters. Cf Data/ folder to see what choice there is.

Teff = 5700          # Effective temperature  [K]
logg = 4.5           # logarithm of the gravity 
vsini = 2            # A measure for the stellar rotation, determines the width of the spectral lines
vmicro = 1           # Micro-turbulent velocity, affects the depth of your spectral lines
FeH = 0.0            # Metallicity: how much metals in the stellar atmosphere. Determines the depth of the spectral lines
Vmag = 9.            # Magnitude of the star (i.e. its brightness) 

exposureTime = 900.

# Load and process the stellar spectrum
# This is for 1 telescope only. Use 4*telescopeArea instead of telescopeArea if the spectrum for 4 telescopes is needed.

R = 100000
spectrum = loadSpectrum2(Teff, logg, vmicro, FeH, vsini, R, effWavelSpectrograph, samplingResolution, velocityShift=3.0)


# Use the following line if you want a S/N ratio throughout the spectrum that is wavelength dependent 

# spectrum = electronFlux(spectrum, telescopeArea, throughputInterpolator, lambdaMin, lambdaMax, exposureTime, Vmag, includePoissonNoise=True, normalizeSpectrum=True)


# Use the following line if you want to have a constant S/N throughout the spectrum

SNratio = 50
spectrum= normalizedSpectrumxWithFixedSN(spectrum, lambdaMin, lambdaMax, SNratio)

# Plot the results

fig, ax = plt.subplots(figsize=(18,9))
ax.plot(spectrum['wavel'], spectrum['NelectronsLine'], c="steelblue")
ax.set_xlabel("Wavelength [nm]")
ax.set_ylabel("Flux [e-/exposure]")
ax.grid(True, color='gainsboro', linestyle='-', linewidth=0.5)
plt.show()


