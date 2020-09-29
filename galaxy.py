import os
import sys
import requests
import urllib
import bs4
import bz2
import warnings
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import logging as log

from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord, Angle
from astropy import units as u
from astropy.utils.exceptions import AstropyWarning
from collections import OrderedDict
from scipy.optimize import curve_fit
import scipy.special as sc
from enum import Enum

# Ignores covariance warning from scipy
sc.seterr(all='ignore')
# Removes the 'RADECSYS deprecated' warning from astropy cutout
warnings.simplefilter('ignore', category=AstropyWarning)
# Override system recursion limit for DFS
sys.setrecursionlimit (10**6)

# Setting default logger for the module
batchlog = log.getLogger (__name__)
batchlog.setLevel(log.INFO)
fileHandler = log.FileHandler("./galaxy.log", mode='w')
fileHandler.setFormatter (log.Formatter ("%(levelname)s : GALAXY : %(asctime)s : %(message)s",
                         datefmt='%m/%d/%Y %I:%M:%S %p'))

# Ensures only one file handler is attached to the default logger
for h in batchlog.handlers :
    batchlog.removeHandler (h)

batchlog.addHandler(fileHandler)
batchlog.info ("Welcome!")

class GalType (Enum) :
    """
    Enum for the final verdict of a galaxy
        INVALID_OBJID       - Self explanatory
        FAIL_DLINK          - Scraping download links for FITS files failed
        FILTERED            - Filtered by the filtration condition
        NO_PEAK             - SGA did not return any peaks
        SINGLE              - Single nuclei galaxy
        DOUBLE              - Double nuclei galaxy
    """
    (INVALID_OBJID,
    FAIL_DLINK,
    FILTERED,
    NO_PEAK,
    SINGLE,
    DOUBLE) = tuple(range(6))

    def __str__(self) :
        """ Enum as string """
        return self.name

class Galaxy () :
    """Contains all of the galaxies info"""

    # Parameters for canny
    cannyLow = 25
    cannyHigh = 50

    # Color for marking the hull
    hullMarker = (0, 0, 255)
    # Color for marking the peaks found by SGA
    peakMarker = (255, 0, 0)
    # Color for marking signal region in hull
    signalMarker = (64, 224, 208)

    # Empty (0,) numpy array
    emptyArray = lambda : np.array([])
    # Empty (0,2) numpy array
    emptyArray2D = lambda : np.array([]).reshape(-1, 2)
    # Empty 2-D image
    emptyImage = lambda : Galaxy.emptyArray2D()
    # Empty list of indices
    emptyInds = lambda : Galaxy.emptyArray2D()
    # Empty list of peaks
    emptyPeaks = lambda : OrderedDict({})

    # Two dimensional multi-indexer
    twoDIndexer = lambda x : (lambda f : (f(x[:,0]), f(x[:,1])))\
                            (lambda y : [] if not y.size else y)
    # Two dimensional indexer
    twoDindex = lambda x : (x[0], x[1])
    # Boolean array for equality within region
    ptRegCheck = lambda pt, reg : (pt == reg).all(axis=1)
    # Check if point is in a region
    isPointIn = lambda pt, reg : Galaxy.ptRegCheck(pt, reg).any()
    # Return index of a point in a region
    ptIndex = lambda pt, reg : np.argwhere(Galaxy.ptRegCheck(pt, reg))
    ######################################################################
    # Returns the 7x7 neighborhood centred around a pixel Any neighborhood
    # point that falls outside the search region is discarded
    ######################################################################
    tolNeighs = lambda pt, reg, t : [(pt[0]+dx, pt[1]+dy)
                                    for dx in range(-t,t+1) for dy in range(-t,t+1)
                                    if (dx != 0 or dy != 0)
                                    and Galaxy.isPointIn((pt[0]+dx, pt[1]+dy), reg)]

    def toThreeChannel (im1) :
        """
        Takes a single channel grayscale image and
        converts it to 3 channel grayscale
        """

        im3 = np.empty (im1.shape + (3, ), dtype=np.uint8)
        for i in [0, 1, 2] :
            im3[:,:,i] = np.copy(im1)
        return im3

    def stripDict (dic, bands) :
        """
        Takes a dictionary attribute of a galaxy object and
        strips it down to contain contain keys in argument 'bands'.
        If 'bands' is the empty string, then the entire dictionary
        is returned
        """

        return dic if not bands else {b:_ for b,_ in dic.items() if b in bands}

    def copyRet (dic, bands, asDict) :
        """
        Repeated condition in diagnosis methods
            dic         - dictionary to be returned
            bands       - bands of the image to be returned
                        If null, return all bands
            asDict      - Whether or not to return as a dictionary
                        Needed for single bands

        The dict is returned in the following cases
            1. 'asDict' is True
            2. 'bands' is the empty string
            3. Number of bands specified is greater than one

        If none of the above is true, then the caller requested a single band
        not in dictionary form. This is subsequently returned
        """

        return dic if asDict or not bands or len(bands) > 1 else dic[bands]

    #############################################################################################################
    #############################################################################################################

    def __init__ (self, objid, cood, fitsFold, bands="ri") :
        """
        Constructor for the galaxy object
            objid       - Object id                 (from .csv file)
            cood        - Coordinates of the object (from .csv file)
            fitsFold    - Directory for FITS files  (Supplied by Batch object)
            bands       - Which bands for object    (Supplied by Batch object)
        """

        (self.objid,                    # Object ID as a string
        self.bands,                     # Bands in (u, g, r, i, z) in which to classify
        self.cood,                      # Celestial coordinates of the object
        self.fitsFold,                  # Folder where the FITS image of the object should be downloaded
        self.repoLink,                  # Repository link of the FITS file
        self.gtype                      # Final classification result (Held as an enum --> Check 'Class GalType')
        ) = objid, bands, cood, fitsFold, None, None

        ######################################################################
        # Initialising dictionaries to None. The convention used is that if
        # one of the dict attributes is None or Empty (check class lambda funcs)
        # then, the dict attributes following it are None, Empty be default

        # This chain is followed until the 'filtrate' attribute. This is because,
        # all the computations beyond that point are very heavy. They'll be done
        # only for bands in which there's a reasonable guarantee of finding
        # a single/double galaxy
        ######################################################################
        (self.downLinks,        # FITS download links in all bands              --> Empty Dict or Full
        self.cutouts,           # Raw cutout data                               --> None signifies FITS Failure
        self.imgs,              # Smoothened cutout image                       --> Empty (0, 2) numpy array signifies no appreciable signal in image / FITS failure

        ######################## Failures cascade down #######################

        self.hullInds,          # Convex hull indices                           --> Empty (0, 2) numpy array signifies no appreciable signal in image
        self.hullRegs,          # Region enclosed by convex hull                --> Empty (0, 2) numpy array signifies tightly enclosed hull
        self.regInfo,           # Pixel intensity count of hull region          --> Same as above, but the hull is all dark

        ########################### Failure WALL ############################

        self.filtrate,          # Condition not to process any further          --> Finding an object/signal is highly improbable

        ######################## Failures cascade down #######################

        self.gaussParams,       # (mean, sigma, noise, half-width half-max)     --> Empty () on filtration, or fitting failure
        self.searchRegs,        # Region in hull above noise                    --> Empty (0, 2) numpy array on fitting failure
        self.noises,            # Noise value for SNR calculation               --> If None, don't use SNR filtration in SGA
        self.gradPeaks,         # Peaks found in SGA                --> Empty OrderedDict for no peaks
        self.finPeaks           # Reduced peak list after dfs                   --> Empty [] for no peaks
        ) = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}

        ######################################################################
        # Cutouts is init to None to signify -
        # 1. FITS file has failed to download/extract
        # 2. FITS file has failed to open
        # 3. Cutout from FITS file has failed
        ######################################################################
        self.cutouts = {b:None for b in bands if b in "ugriz"}

        batchlog.info ("{} --> Initialised".format(self.objid))

    def __str__ (self) :
        """ Galaxy object to string """
        return self.objid

    def scrapeRepoLink (self) :
        """
        Set the FITS repository link (for all bands)
        """

        link = "http://skyserver.sdss.org/dr14/en/tools/explore/summary.aspx?objid=" + self.objid
        try :
            soup = bs4.BeautifulSoup(requests.get(link).text, features='lxml')
        except Exception as e :
            batchlog.error("{} --> Error in obtaining repository link".format(self.objid))
            raise e

        # Condition for invalid link
        if len(soup.select(".nodatafound")) == 1 :
            self.repoLink = None
            return

        tagType = type(bs4.BeautifulSoup('<b class="boldest">Extremely bold</b>' , features = 'lxml').b)
        for c in soup.select('.s') :
            tag = c.contents[0]
            if tagType == type(tag) and (fitsLinkTag:=str(tag)).find('Get FITS') > -1 :
                break

        fitsLinkTag = fitsLinkTag[fitsLinkTag.find('"')+1:]
        fitsLinkTag = fitsLinkTag[:fitsLinkTag.find('"')].replace("amp;",'')
        self.repoLink = "http://skyserver.sdss.org/dr15/en/tools/explore/" + fitsLinkTag

    def scrapeBandLinks (self) :
        """Computes it list of type [(Band, download link for band)] for
        all bands in 'ugriz' and sets it to object attribute"""

        def procClass (st) :
            """Helper function to extract download link"""
            st = st[st.find('href='):]
            st = st[st.find('"')+1:]
            return st[:st.find('"')]

        try :
            dlinks = {(lambda s: s[s.rfind('/') + 7])(procClass(str(x))) : procClass(str(x))
            for x in
            bs4.BeautifulSoup(requests.get(self.repoLink).text, features = 'lxml').select(".l")[0:5]
            }
        except Exception as e :
            batchlog.error("{} --> Error in obtaining band download links".format(self.objid))
            raise e

        self.downLinks = dlinks

    def downloadExtract (self, b, dlink) :
        """
        For a given band 'b', downloads the .bz2 container of the FITS file
        from the argument 'dlink' and extract it
        """

        # Download path for the .bz2
        dPath = os.path.join(self.fitsFold, dlink[dlink.rfind('/')+1:])

        try :
            # Downloading .bz2 to 'dPath'
            urllib.request.urlretrieve(dlink, dPath)
        except Exception as e :
            batchlog.error("{} --> Error in obtaining .bz2 for {} band".format(self.objid, b))
            raise e

        try :
            zipf = bz2.BZ2File(dPath)
            data = zipf.read()
            zipf.close()
            extractPath = dPath[:-4]

            ######################################################################
            # Extracts the .bz2, deletes the archive and renames FITS file to
            # {objid}-{band}.fits
            ######################################################################
            open(extractPath, 'wb').write(data) ;
            os.rename(extractPath, self.getFitsPath(b))
            os.remove(dPath)
        except Exception as e :
            batchlog.error("{} --> Error in extraction of .bz2 for {} band".format(self.objid, b))
            raise e

    def download (self) :
        """
        Downloads the frame for an object id for all bands.
        1. If repository link is None, then the object ID is invalid
        2. If the download links dict is empty, then scraping the links has failed
        """

        # Bands that remain to be downloaded. Ignores invalid bands in 'band' attribute
        toDown = [b for b in self.bands if b in "ugriz" and not os.path.exists(self.getFitsPath(b))]
        batchlog.debug ("toDown = {}".format(toDown))
        if not toDown :
            batchlog.info ("{} --> FITS files of all bands already downloaded".format(self.objid))
            return
        batchlog.info ("{} --> FITS bands to be downloaded - {}".format(self.objid, toDown))

        # Initialised at constructor
        if self.repoLink is None :
            self.scrapeRepoLink()
            if self.repoLink is None :
                self.gtype = GalType(GalType.INVALID_OBJID)
                batchlog.warning("{} --> Set gtype to INVALID_OBJID".format(self.objid))
                return

        batchlog.info("{} --> FITS repository link successfully retrieved".format(self.objid))

        # Initialised at constructor
        if not self.downLinks :
            self.scrapeBandLinks()
            if not self.downLinks :
                self.gtype = GalType(GalType.FAIL_DLINK)
                batchlog.warning("{} --> Set gtype to FAIL_DLINK".format(self.objid))
                return

        batchlog.info("{} --> FITS bands download links retrieved".format(self.objid))

        for b, dlink in self.downLinks.items() :
            if b in self.bands and b in "ugriz" :
                # Download only if the FITS file doesn't exist
                if not os.path.exists (self.getFitsPath(b)) :
                    self.downloadExtract (b, dlink)
                batchlog.info("{} --> Obtained FITS file for {}-band".format(self.objid, b))

    def cutout (self, rad=40) :
        """
        Performs a cutout centred at attribute 'cood' for radius 'rad'
        for all bands
        """

        def cutout_b (fitsPath) :
            """ Helper function that performs cutout for a band
            for a given radius in argument 'rad' """

            # In case the FITS file was not downloaded due to any error
            if not os.path.exists (fitsPath) :
                batchlog.warning ("{} --> No cutout performed as FITS file doesn't exist".format(self.objid))
                return None

            try :
                hdu = fits.open (fitsPath, memmap=False)[0]
            except Exception as e :
                batchlog.error("{} --> Error in loading FITS file for {}-band".format(self.objid, b))
                raise e

            wcs = WCS(hdu.header)
            position = SkyCoord(ra=Angle (self.cood[0], unit=u.deg),
                            dec=Angle (self.cood[1], unit=u.deg))
            size = u.Quantity ((rad, rad), u.arcsec)
            return Cutout2D (hdu.data, position, size, wcs=wcs).data

        for b in self.bands :
            ######################################################################
            # Considers valid bands only and does cutout only if dict[band] is
            # still None as I/O is costly
            ######################################################################
            if b in "ugriz" and self.cutouts[b] is None :
                self.cutouts[b] = cutout_b (self.getFitsPath(b))
            batchlog.info("{} --> Got cutout for {}-band".format(self.objid, b))

    def smoothen (self, reduc=2, sgx=5, sgy=5) :
        """
        Performs a smoothening on the raw cutout data as follows (for each band) -
            reduc       - Factor by which the median of the cutout image
                        will be reduced before it is considered for the
                        'vmin' argument to LogNorm
            sgx         - Sigma_x factor for gaussian kernel
            sgy         - Sigma_y factor for gaussian kernel

        The smoothening process works as follows -
            1. Push up negative pixels to 0
            2. Subtract the minimum of all positive pixels from all
            positive pixels
            3. After extensive observation of the LogNorm function, it was
            noted that 0.1 gives reasonable scaling for 'r' band objects.
            This is compared with the reduced median to set 'vmin'. 'vmax'
            is set as the maximum of the image
                -> If 'vmin' >= 'vmax', LogNorm will behave erratically
                -> This physically means that there is no proper distinction
                between noise and signal in this image
                -> Hence discard
            4. Perform the  LogNorm with clipping parameter set to true
            5. Push up masked values to 0
            6. Scale the resulting image within the grayscale range [0, 255]
            7. Convolve with the gaussian kernel over this image
                -> Helps to smoothen masked points
        """

        def smoothen_b (img) :
            """ Helper function for that performs smoothening for a given
            band. Arguments are described in enclosing method """

            # Step 1
            img[img < 0] = 0
            # Step 2
            img[img >= 0] -= np.min(img[img >= 0])
            # Step 3
            if (vmin:=max(0.1, np.median(img)/reduc)) >= (vmax:=np.max(img)) :
                batchlog.info("{} --> Appreciable intensity not found while smoothing".format(self.objid))
                return Galaxy.emptyImage()
            # Step 4
            imgNorm = colors.LogNorm(vmin, vmax, True).__call__(img)
            # Step 5
            imgNorm.data[imgNorm.mask] = 0

            ######################## Step 7 ################## Step 6 #####################
            return (lambda d:cv2.GaussianBlur(np.floor(255*(lambda x,mn,mx : (x-mn)/(mx-mn))
                                                           (d, np.min(d), np.max(d))
                                                    ).astype(np.uint8), (sgx,sgy),0
                    ))(imgNorm.data)

        for b, cut in self.cutouts.items() :
            self.imgs[b] = smoothen_b(cut)
            batchlog.info("{} --> Smoothened {}-band".format(self.objid, b))

    def hullRegion (self) :
        """
        Finds the region where SGA has to be performed for each band
        Works as follows -
            1. Uses canny with parameters (defined as Class variables)
            to find edges
            2. Takes the convex hull of these edges to find the boundary
                a. cv2.convexHull actually returns the set of minimal points
                needed to define the contour
                b. cv2.drawContours uses this to draw the actual boundary on
                an image
                c. The boundary is then found by searching for the boundary
                color marker (Galaxy.hullMarker)
            3. Scans the boundary from left to right to find the region
            it contains
                -> This works due to the 'convex' property of the hull
                ->  For any two points in the region contained in the hull,
                    there is a straight line that connects them which exists
                    entirely within the hull (Definition of convexity)
        """

        def hullRegion_b (img) :
            """ Takes as input the smoothened image of a band and returns
            the hull boundary and the region contained between said boundary
            'img' is never None when this function is called """

            # Step 1
            # Lambda function for finding edges returned by Canny
            cannyEdges = (lambda s,l,h : np.argwhere(cv2.Canny(s, l, h) == 255))\
                        (img, Galaxy.cannyLow, Galaxy.cannyHigh)

            # There are no well defined edges if 'cannyEdges' is empty
            if not cannyEdges.size :
                return Galaxy.emptyInds(), Galaxy.emptyInds()

            # Step 2a
            cannyHull = [cv2.convexHull(np.flip(cannyEdges, axis=1))]
            # Step 2b
            fullImg = cv2.drawContours (Galaxy.toThreeChannel(img),
                                    cannyHull,
                                    0, Galaxy.hullMarker, 1, 8)
            # Step 2c
            hullInds = np.argwhere((fullImg == np.array(Galaxy.hullMarker)).all(axis=2))

            # Step 3
            hullRegs = (lambda f, uq : np.array([
                [f[uq[i],0],y]
                for i in range(0, len(uq))
                    for y in (lambda ys:np.setdiff1d(range(np.min(ys), np.max(ys)+1), ys))\
                            (f[uq[i]:,1]
                            if i == len(uq)-1
                            else f[uq[i]:uq[i+1],1])
                                        ])
            )(*(lambda f : (f, np.unique(f[:,0], True)[1]))\
                (hullInds))

            return hullInds, Galaxy.emptyInds() if not hullRegs.size else hullRegs

        for b, img in self.imgs.items() :
            self.hullInds[b], self.hullRegs[b] = (Galaxy.emptyInds(), Galaxy.emptyInds()) if not img.size \
                                                else hullRegion_b(img)
            batchlog.info("{} --> Obtained hull boundary and region for {}-band".format(self.objid, b))

    def distInfo (self) :
        """
        Computes the frequency of pixel values in the
        hull region for each band
        """

        def calcInfo (series) :
            """ Helper function that returns the count info of the data """

            uq = np.unique(series, return_counts=True)

            ######################################################################
            # grays - Contains the grayscale values in the image
            # counts - Number of pixels with a particular grayscale value
            ######################################################################
            grays, counts = tuple(np.split(np.array([
                                        [i, uq[1][ag][0]]
                                        for i in range(0,256)
                                        if (ag:=np.argwhere(uq[0]==i).flatten()).size > 0
                                        ]), 2, axis=1))
            return grays, counts

        for b, series in self.getSmoothRegSeries().items() :
            self.regInfo[b] = (Galaxy.emptyArray(), Galaxy.emptyArray()) if not series.size else calcInfo (series)
            batchlog.info("{} --> Set the hull region count info for {}-band".format(self.objid, b))

    def filter (self, per=10) :
        """
        Filtration condition before classification -
            1. Cutout is None                                       --> FITS failure
            2. Smoothed image is empty                              --> No appreciable signal that can be processed well by LogNorm
            3. Hull boundary is empty                               --> No sharp object in the image, mostly dark
            4. Hull region is empty                                 --> Object too small to conduct SGA
            5. Centre is not in hull region                         --> Hull encloses an object with a different SDSS objID
            6. Hull region                                          --> Region encloses more than (100 - per)% of the image. Signifies very noisy image
        If true, do not classify. Finding an object/signal is highly improbable
        """

        for b in self.bands :
            if b in "ugriz" :
                self.filtrate[b] = self.cutouts[b] is None or\
                        not self.imgs[b].size or\
                        not self.hullInds[b].size or\
                        not self.hullRegs[b].size or\
                        not Galaxy.isPointIn ((self.imgs[b].shape[0]//2, self.imgs[b].shape[1]//2), self.hullRegs[b]) or\
                        (lambda x,y : 100*(x-y)/x < per)(self.imgs[b].flatten().shape[0], self.hullRegs[b].shape[0])
                batchlog.info("{} --> Calculated filtrate for {}-band".format(self.objid, b))

    def fitGaussian (self, noiseSig=10, hwhmSig=2) :
        """
        Fit a gaussian to the inverse grayscale intensity histogram
        and deduce the noise/signal cutoff. Background -
        1. Assumes a 2-D radial gaussian profile of the intensity distribution
        2. Count per grayscale value is a linear function of the radial distance
        from the maximum intensity point
            -> More specifically, we have assumed count = 2*pi*r
            -> Maximum intensity point may be due to a stray object
            -> Hence, this is not fixed and is deduced by curve fitting
        3. Fits the counts vs. grays scatter plot as a gaussian of unknown mean and variance
        4. Computes the noise level at 1/10th the mean
        5. Half-width-half-maximum is used as the lower threshold for the pixel value of a peak
        """

        def fitGaussian_b (info) :
            """ Helper function that fits the gaussian for each band """

            grays, counts = info
            ming = 1 if not np.min(grays) else 0
            x, y = grays[ming:], counts[ming:]
            gaussian = lambda z, gp, sg : gp*np.exp(-np.square(z/(2*np.pi*sg)))

            try :
                gaussPeak, sigma = curve_fit (gaussian, y.flatten(), x.flatten(), p0=[np.max(x), np.max(y)/20])[0]
            except :
                batchlog.warning ("Fault in curve_fit")
                return ()

            return gaussPeak, sigma, np.floor(gaussPeak)/noiseSig, np.floor(gaussPeak/hwhmSig)

        for b, filt in self.filtrate.items() :
            self.gaussParams[b] = () if filt else fitGaussian_b(self.regInfo[b])
            batchlog.info("{} --> Fit gaussian for {}-band".format(self.objid, b))

    def searchRegion (self) :
        """
        For each band, finds the region within the hull that is above the noise
        level determined by gaussian fitting. This is where SGA
        will be performed
        """

        ######################################################################
        # Helper function that finds the search region in the hull region
        # based on the cutoff noise
        ######################################################################
        searchReg_b = lambda hr, img, ns : np.array([
            pt for pt in hr if img[Galaxy.twoDindex(pt)] >= ns
        ])

        for b,filt in self.filtrate.items() :
            self.searchRegs[b] = Galaxy.emptyInds() if filt or not self.gaussParams[b]\
                                else searchReg_b(self.hullRegs[b], self.imgs[b], self.gaussParams[b][2])

    def sga (self, reduc=100, tol=3) :
        """
        Performs Stochastic Gradient Ascent (SGA) in the region indicated by attribute 'hullReg'
        for each band. It works as follows -
            1. Initialises a point randomly in the region
            2. In a 7x7 region, finds the pixel which has the
            highest value greater than the current pixel
                -> Note that 7x7 is to be interpreted as (2*tol + 1) x (2*tol + 1)
                at every piece of documentation in this method
                -> 'tol' defines the no. of neighbors to be considered during
                SGA
            3. Sets the current pixel to this max pixel
            4. Repeat the above steps until no greater pixel can be found in the
            7x7 region

        The above routine is performed 'iter' times (defined in internal function
        gradAsc_b). The final peak (pixel coordinate) of each run of SGA
        is compared with a running list of peaks and a filtration condition is applied
        to include it or not. This condition is described in detail in the comments

        'reduc' is a proportionality factor to decide the number of runs of
        SGA to perform
        """

        def sga_b (searchReg, gradKey, hwhm) :
            """ Performs SGA on a region for a particular band
            Argument 'gradKey' is a lambda function for the underlying pixel
            values for the band image """

            batchlog.debug("Size of region = {}".format(len(searchReg)))
            if not searchReg.size :
                return Galaxy.emptyPeaks()

            ######################################################################
            # The runs of SGA is taken to be proportional to the
            # length of the region
            ######################################################################
            iters = max(len(searchReg)//reduc, 100)

            # Running peak list. {(x,y) : [pixel value, frequency]}
            peaks = {}

            ######################################################################
            # Filtration condition -
            # 1. No other neighbor in the 7x7 neighborhood of the peak must already
            # be in the peak list
            # 2. All the points in a 1x1 region around the peak must be in the
            # search region
            #   -> Otherwise the peak found is too close to the edge
            # 3. The peak found must be above the half-width half-maximum level
            #
            # Each line in the lambda function below represents the conditions
            # in the above order
            ######################################################################
            peakFilter = lambda pk :\
            not np.array([x in peaks for x in Galaxy.tolNeighs(pk, searchReg, tol)]).any() and\
            len(Galaxy.tolNeighs(pk, searchReg, 1)) == 8 and\
            gradKey(pk) >= hwhm

            # No need to keep track of iteration number
            for _ in range(0, iters) :
                # Step 1
                st = pk = pt = (lambda x:(x[0], x[1]))(searchReg[np.random.choice(range(0, searchReg.shape[0]))])
                while pt :
                    ######################################################################
                    # If at any point in the interation, the list of neighboring 7x7 points
                    # is empty, it means that the point has gotten trapped somewhere. This
                    # can occur when the region under consideration is miniscule
                    ######################################################################
                    if not (ns:=Galaxy.tolNeighs(pt, searchReg, tol)) :
                        break

                    # Step 2
                    mxn = max(ns, key = gradKey)
                    # Step 3
                    pt = mxn if gradKey(mxn) > gradKey(pk:=pt) else None # Step 4 (Termination)

                # If peak already exists, increase its frequency count
                if pk in peaks :
                    peaks[pk][1] += 1
                # If peakFilter allows the peak, create a new entry in the dict
                elif peakFilter(pk) :
                    peaks[pk] = [gradKey(pk), 1]

            return OrderedDict (sorted (peaks.items(), key=lambda x:x[1][0], reverse=True))

        for b, filt in self.filtrate.items() :
            self.gradPeaks[b] = Galaxy.emptyPeaks() if filt or not self.searchRegs[b].size\
                            else gradAsc_b(self.searchRegs[b],
                                        lambda x:self.imgs[b][x],
                                        self.gaussParams[b][3])

            batchlog.info("{} --> Found search region and peak list for {}-band".format(self.objid, b))

    def noiseLevel (self) :
        """ For each band, computes the average noise in the image, which will
        be used in computing the SNR of peaks during classification """

    def classify (self) :
        """
        Final classification of the image. Works as follows
        1. If gtype is already set, do not perform SGA
            -> INVALID_OBJID or FAIL_DLINK
        2. If filtered, do not perform SGA
        3. Find the raw peaks returned by gradient ascent
            -> It has its own parsimonious filtering condition
        4.
        """

        ######################################################################
        # Returns the indices of neighboring points (specified by index 'ind')
        # in a region 'reg'
        ######################################################################
        dfsNeighs = lambda ind, reg : [ni[0][0] for pt in
                                    Galaxy.tolNeighs(reg[ind], reg, 1)
                                    if (ni:=Galaxy.ptIndex(pt, reg)).size > 0]

        def dfs (ind, reg, vis, comp) :
            # Mark current index as visited
            vis[ind] = True

            # Append current index to the present component that is being created
            comp.append (ind)

            # For each neighbor, if it hasn't already been visited, visit it
            for i in dfsNeighs (ind, reg) :
                if not vis[i]:
                    dfs (i, reg, vis, comp)

        def connComps (reg) :
            noPts = len(reg)
            vis = np.repeat(False, noPts)
            comps = []
            seed = 0

            # Seed indices that each returns one connected component
            while seed < noPts :
                dfs(seed, reg, vis, comp:=[])

                # Mapping indices to mixel coordinates
                comps.append(np.array([
                    reg[i] for i in comp
                ]))

                # Find the next seed index, if it exists
                while seed < noPts and vis[seed] :
                    seed += 1

            # Return the list of components as a list. Each entry is a list of pixel coordinates
            return comps

        def filterPeaks (peaks, sreg) :
            """ Helper function for filtering gradient ascent peaks """

            # No peaks detected
            if not peaks :
                return []
            if len(peaks) == 1 :
                return []

            # Obtain connected components
            comps = connComps(sreg)

            # Mapping which component each peak belongs to
            pk_comp = {p:i for i, c in enumerate(comps) for p in peaks if Galaxy.isPointIn(p, c)}

            # Acquiring the top two brightest peaks
            top = list(peaks)[:2]
            p1, p2 = top[0], top[1]

            # If part of the same component then return both
            if pk_comp[p1] == pk_comp[p2] :
                return top
            # Otherwise return the one which belongs to the larger connected component
            else :
                return [p1 if len(comps[pk_comp[p1]]) > len(comps[pk_comp[p2]]) else p2]


        for b, gp in self.gradPeaks.items() :
            # gtype could be set to INVALID_OBJID or FAIL_DLINK from the downloading phase
            if self.gtype is not None :
                self.finPeaks = {b:[] for b in self.bands if b in "ugriz"}
                continue

            # If filter, final peaks in each band is an empty list
            self.finPeaks[b] = [] if self.filtrate[b] or not gp \
                                else filterPeaks (gp, self.searchRegs[b])

        # Mapping filtered peak list in each band to its length
        cntZero, cntOne, cntTwo = 0, 0, 0
        for l in [len(fp) for _, fp in self.finPeaks.items()] :
            if l == 0 :
                cntZero += 1
            elif l == 1 :
                cntOne += 1
            else :
                cntTwo += 1

        # Self-explanatory
        if cntTwo :
            self.gtype = GalType(GalType.DOUBLE)
        elif cntOne :
            self.gtype = GalType(GalType.SINGLE)
        else :
            self.gtype = GalType(GalType.NO_PEAK)

    #############################################################################################################
    #############################################################################################################

    def getFitsPath (self, b) :
        """
        Helper function for an object that returns the FITS file path
        for the band specified by argument 'b'
        """

        return os.path.join (self.fitsFold, "{}-{}.fits".format(self.objid, b))

    def getCutout (self, bands="", asDict=False) :
        """
        Returns a copy of the cutout of the specified band(s)
        If asDict is set,
            then images are returned as a dictionary
        Else,
            it's returned as a dictonary only if there are multiple bands
            otherwise,
                the image of the band is returned

        Additionally, if triple is set, then it is the three-channel copy of
        the image that is returned
        """

        cuts = {b:Galaxy.emptyImage() if cut is None else cut
            for b,cut
            in Galaxy.stripDict(self.cutouts, bands).items()}
        return Galaxy.copyRet(cuts, bands, asDict)

    def getCutoutSeries (self, bands="", asDict=False) :
        """ Flattens the cutout image """

        cutFlat = {b:cut.flatten()
                for b, cut in self.getCutout(bands, True).items()}
        return Galaxy.copyRet(cutFlat, bands, asDict)

    def getCutoutRegSeries (self, bands="", asDict=False) :
        """ Flattens the hull region for cutout image """

        cutRegFlat = {b:cut[Galaxy.twoDIndexer(self.hullRegs[b])].flatten()
                    for b, cut in
                    self.getCutout(bands, True).items()}
        return Galaxy.copyRet(cutRegFlat, bands, asDict)

    def getSmooth (self, bands="", triple=False, asDict=False) :
        """ Returns the smoothed image """

        imgs = {b:Galaxy.toThreeChannel(np.zeros_like(self.cutouts[b]) if not img.size else img) if triple
            else np.copy(img)
            for b, img in Galaxy.stripDict(self.imgs, bands).items()}

        return Galaxy.copyRet(imgs, bands, asDict)

    def getSmoothSeries (self, bands="", asDict=False) :
        """ Flattens the smoothed image """

        imgFlat = {b:img.flatten()
                for b, img in
                self.getSmooth(bands, False, True).items()}

        return Galaxy.copyRet(imgFlat, bands, asDict)

    def getSmoothRegSeries (self, bands="", asDict=False) :
        """ Flattens the hull region for smoothed image"""

        smoothRegFlat = {b:img[Galaxy.twoDIndexer(self.hullRegs[b])].flatten()
                        for b,img in
                        self.getSmooth(bands, False, True).items()}

        return Galaxy.copyRet(smoothRegFlat, bands, asDict)

    def getHullMarked (self, bands="", sig=False, asDict=False) :
        """ Returns a copy of the smoothed image of the specified band(s) with hull marked"""

        hullMarks = self.getSmooth(bands, True, True)
        for b, hi in Galaxy.stripDict(self.hullInds, bands).items() :
            img = hullMarks[b]
            img[Galaxy.twoDIndexer(hi)] = np.array(Galaxy.hullMarker)
            if sig :
                img[img[...,0] > self.gaussParams[b][2]] = np.array(Galaxy.signalMarker)

        return Galaxy.copyRet(hullMarks, bands, asDict)

    def getGradPeaksMarked (self, bands="", hull=False, asDict=False) :
        """
        Returns a copy of the smoothed image of the specified band(s) with peaks marked
        Additionally, if hull is set, then the hull is marked as well
        """

        peakMarks = self.getHullMarked(bands, False, True) if hull else self.getSmooth(bands, True, True)
        for b,pks in Galaxy.stripDict(self.gradPeaks, bands).items() :
            for pk in pks :
                peakMarks[b][pk] = np.array(Galaxy.peakMarker)

        return Galaxy.copyRet(peakMarks, bands, asDict)

    def getHistAxes (info, constrictHist, invHist) :
        """ Helper function to return the axes for the plot """

        grays, counts, ming, maxg = info
        freq = np.zeros(256)
        freq[[] if not grays.size else grays] = counts
        maxg += 1 if not invHist and maxg < 255 else 0

        ret = (np.arange(ming, maxg, 1), freq[ming:maxg])
        return ret[-1::-1] if invHist else ret

    def getRegHist (self, bands="", constrictHist=False, invHist=False, asDict=False) :
        """ Returns (x, y) hull region histogram data for all bands """

        regHists = {b:Galaxy.getHistAxes(info, constrictHist, invHist)
                for b, info
                in Galaxy.stripDict(self.regInfo, bands).items()}

        return Galaxy.copyRet(regHists, bands, asDict)

    def getGaussPlot (self, bands="", asDict=False) :
        """
        Returns the axes data for the inverse intensity scatter,
        fit gaussian curve and cutoff level
        """

        def getGaussArgPack (info, cutoff, divs=1000) :
            """ Helper function to return (*args, **kwargs), to be directly called
            by matplotlib for plotting pixel intensity scatter and gaussian fit """

            x, y = info
            batchlog.debug ("{}".format(x))
            l = 1 if not np.min(x) else 0
            x, y = x[l:], y[l:]

            gpeak, sigma, noise = cutoff[:-1]
            lin = np.linspace(np.min(y), np.max(y), divs)
            linGauss = (lambda z:gpeak*np.exp(-np.square(z/sigma)))(lin)

            return [{"args":(y, x, 'o'), "kwargs":{'markersize':3}},
            {"args":(lin, linGauss, 'r'), "kwargs":{}},
            {"args":(lin, noise*np.ones_like(lin), '--k'), "kwargs":{}}]

        argPacks = {b:getGaussArgPack(self.regInfo[b], self.gaussParams[b])
                    for b, filt in self.filtrate.items() if not filt}

        return Galaxy.copyRet(argPacks, bands, asDict)
