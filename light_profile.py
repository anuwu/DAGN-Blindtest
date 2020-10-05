import numpy as np
import warnings
import logging
import scipy.special as sc
from scipy.optimize import curve_fit
from scipy.optimize.minpack import OptimizeWarning
from scipy.integrate import cumtrapz as cint

import plane_coods as pc

# Ignores covariance warning from scipy
warnings.simplefilter('ignore', category=OptimizeWarning)

# Setting the logger
log = logging.getLogger(__name__)
log.setLevel(logging.WARNING)
fileHandler = logging.FileHandler("./light_profile.log", mode='w')
fileHandler.setFormatter(logging.Formatter("%(levelname)s : LIGHT_PROFILE : %(asctime)s : %(message)s",
                         datefmt='%m/%d/%Y %I:%M:%S %p'))

# Ensures only one file handler
for h in log.handlers :
    log.removeHandler(h)

log.addHandler(fileHandler)
log.info("Welcome!")

class LightProfile () :
    """
    Stores a grayscale pixel distribution and contains methods to infer
    the noise/signal level from it depending on a particular fitting curve
        1. Parameters of the fit
        2. Noise level
        3. Signal level
    """

    emptyArgs = 4*[{"args":(), "kwargs":{}}]

    def __seriesToDist__ (self, series) :
        """ Computes the distribution from the flattened data """

        uq = np.unique(series, return_counts=True)

        ######################################################################
        # grays - Contains the grayscale values in the image
        # counts - Number of pixels with a particular grayscale value
        ######################################################################
        grays, counts = tuple(np.split(np.array([
                                    [i, uq[1][np.argwhere(uq[0]==i).flatten()][0]]
                                    for i in range(0,256)
                                    if np.argwhere(uq[0]==i).flatten().size > 0
                                    ]), 2, axis=1))

        ming = 1 if not np.min(grays) else 0
        self.grays, self.counts = grays[ming:].flatten(), counts[ming:].flatten()

    def __init__ (self, series) :
        """
        Constructor for the distribution object - Takes in the hull region
        grayscale values in flattened format
        """

        # Finds pixel distribution from flattened data
        self.__seriesToDist__(series)
        self.initParams = []                                # Initial parameters to seed the fitting function
        self.params = ()                                    # Parameters of the fit, specific to fitting curve
        self.cov = None                                     # Covariance of the fit
        self.noise = None                                   # Noise level derived from the fit
        self.signal = None                                  # Signal level derived from the fit
        self.curveToFit = None                              # Curve to fit, takes as specific input point and parameters as input
        self.fitCurve = None                                # Unary function after fitting
        self.noiseSNR = None                                # Noise from raw cutout data to be used for SNR calculations

    def __str__ (self) :
        """ Fit parameters to string """
        return "Parameters = {}".format(self.params)

    def fit (self) :
        """ Generic code to perform the fitting """
        self.params, self.cov = curve_fit(self.curveToFit, self.counts, self.grays, p0=self.initParams, maxfev=5000)


        # Partially apply the fit parameters to expose the unary fit function
        self.fitCurve = self.fitCurve(*self.params)
        log.info("Fit done --> {}".format(self))

    def estimateNoise (self, cutout, smooth, iters=100) :
        """
        Computes the noise intensity that will be used in SNR calculations for peak reporting
            cutout          - The raw cutout image whose pixel values are used to
                            average the noise
            smooth          - Smoothed image from which the coordinates of noise pixels
                            will be found
            iters           - Number of samplings over which the noise is averaged

        Works as follows -
            1. Computes the coordinates that are below the signal
            2. Randomly chooses a coordinate
            3. Averages pixel values over its neighbors
            4. Repeat the above 2 steps 100(iter) times and average
        """

        # If there is no pixel below noise level, then don't calculate it
        noiseCoods = np.argwhere(smooth < self.noise)
        if not noiseCoods.size :
            log.info("No coordinates where noise could be found. Retain snrNoise as None")
            return

        noiseTot = 0
        for _ in range(0, iters) :
            pt = tuple(noiseCoods[np.random.choice(range(0, len(noiseCoods)))])
            noiseTot += np.mean([
                cutout[p] for p in
                ([pt] + pc.neighsInReg(pt, noiseCoods, 1))
            ])

        self.noiseSNR = noiseTot/iters
        log.info("Estimated noise from cutout data as {}".format(self.noiseSNR))

    def plotArgs (self, divs=10000) :
        """
        Returns args and kwargs for matplotlib to plot the following -
            1. Scatter plot of grayscale intensity distribution
            2. Continuous fit of the light profile
            3. Noise level
            4. Signal level
        """

        lin = np.linspace(np.min(self.counts), np.max(self.counts), divs)
        return [{"args":(self.counts, self.grays, 'o'), "kwargs":{'markersize':3}},
        {"args":(lin, self.fitCurve(lin), 'c'), "kwargs":{}},
        {"args":(lin, self.noise*np.ones_like(lin), '--r'), "kwargs":{}},
        {"args":(lin, self.signal*np.ones_like(lin), '--g'), "kwargs":{}}]


class Gaussian (LightProfile) :
    """
    Parameters and method to infer noise/signal
    for a Gaussian intensity profile
    """

    def __init__ (self, series) :
        """
        Constructor for the gaussian profile object
        I(z) = mean * exp(-(z/4sigma)^2)
        mean, sigma are to be fit
        """

        LightProfile.__init__(self, series)
        self.curveToFit = lambda z, mean, sigma : mean*np.exp(-np.square(z/(4*sigma)))
        self.fitCurve = lambda *params : lambda z : params[0]*np.exp(-np.square(z/(4*params[1])))
        self.initParams = [np.max(self.grays), np.max(self.counts)/20]

        log.info("Initialised Gaussian profile")

    def __str__(self) :
        """ Returns the Gaussian parameters as a string """
        return "(mean, sigma) = {}".format(self.params)

    def inferLevels (self, noiseSig=10, hwhmSig=2) :
        """
        Infers the noise and signal from the noise significance value
        and the half-width-half-maximum significance value. Obtained by
        a simply dividing the significance values with the mean of the gaussian
        """

        self.noise = np.floor(self.params[0]/noiseSig)
        self.signal = np.floor(self.params[0]/hwhmSig)
        log.info("Inferred noise and signal levels for Gaussian profile as {} and {}".\
                format(self.noise, self.signal))


class Sersic (LightProfile) :
    """
    Parameters and method to infer noise/signal for a Sersic intensity profile.
    Galaxy light curves are usually fit with a Sersic profile with an index ranging
    between 0.5 and 10. For this Class, the limits are present in the Class values
    llim = 0.25, and rlim = 15. If the fit sersic index does not fall within this
    range, it is recommended to use a Gaussian profile
    """

    # Lower sersic index limit
    llim = 0.25
    # Upper sersic index limit
    rlim = 15

    def __init__ (self, series) :
        """
        Constructor for the sersic profile object -
            I(z) = I0 - k*z^(1/n)
        k, n are to be fit
        I0 is the maximum value of the data to be fit
        """

        LightProfile.__init__(self, series)
        self.curveToFit = lambda z, k, n : np.max(self.grays) - k*np.power(z/4, 1/n)
        self.fitCurve = lambda *params : lambda z : np.max(self.grays) - params[0]*np.power(z/4, 1/params[1])
        self.initParams = [np.max(self.grays)*np.power(4/np.max(self.counts), 1/4), 1/4]

        log.info("Initialised Sersic profile")

    def __str__(self) :
        """ Returns the Sersic parameters as a string """
        return "(k, n) = {}".format(self.params)

    def inferLevels (self, noisePer=0.95, sigPer=0.5, divs=10000) :
        """
        Arguments -
            noisePer            - Percentage of the integrated light beyond which
                                signal is taken to be absent
            sigPer              - Half integrated light level
            divs                - Number of divisions needed in the numerical integration

        How it works -
            1. Find the supposed frequency at which grayscale value becomes zero
                -> This is needed because we need to cut the fit at the point which
                grayscale value becomes negative, which is unphysical
                -> Call this value fm
            2. Integrate the curve from 0 to fm as the total area
            3. Since the fit is monotonically decreasing, find the point where the
            integrated area is 95%(noisePer) of the total area
                -> This is interpreted as the noise level
            4. Do the above for 50%(sigPer)
                -> This is interpreted as the signal level
        """

        k, n = self.params
        fmax = 4*np.power(np.max(self.grays)/k, n)

        xs = np.linspace(0, fmax, 10000)
        integrand = self.fitCurve(xs)
        intProf = cint(integrand, xs, initial=0)

        def perInd (per) :
            """ Helper function that computes the grayscale level at a given
            integrated light percentage """
            l = len(intProf)
            level = per*intProf[-1]
            ind = np.argmax(np.where(intProf - level < 0,
                                    range(0, l),
                                    -1))
            return integrand[ind]

        self.noise, self.signal = perInd(noisePer), perInd(sigPer)
        log.info("Inferred noise and signal levels for Sersic profile as {} and {}".\
                format(self.noise, self.signal))
