import os
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
from enum import Enum

# Removes the 'RADECSYS deprecated' warning from astropy cutout
warnings.simplefilter('ignore', category=AstropyWarning)

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

class Galaxy () :
    """Contains all of the galaxies info"""

    class GalType (Enum) :
        """Enum for the final verdict of a galaxy
            INVALID_OBJID       - Self explanatory
            NO_NOISE            - Noise level could not be reasonably found from the cutout
            NO_CENTRE           - Centre of the cutout was not inside the hull region
            NO_PEAK             - Gradient ascent did not return any peaks
            SINGLE              - Single nuclei galaxy
            DOUBLE              - Double nuclei galaxy """
        (INVALID_OBJID,
         NO_NOISE,
         NO_CENTRE,
         NO_PEAK,
         SINGLE,
         DOUBLE) = tuple(range(6))

    # Parameters for canny
    cannyLow = 25
    cannyHigh = 50
    hullMarker = (0, 0, 255)
    # Color for marking the peaks found by gradient ascent
    peakMarker = (255, 0, 0)
    # Empty (0,2) numpy array
    __emptyArray2D__ = lambda : np.array([]).reshape(-1, 2)
    # Empty 2-D image
    emptyImage = lambda : Galaxy.__emptyArray2D__()
    # Empty list of indices
    emptyInds = lambda : Galaxy.__emptyArray2D__()
    # Empty list of peaks
    emptyPeaks = lambda : OrderedDict({})
    # Empty series of pixels
    emptySeries = lambda : np.array ([])
    # Two dimensional indexer
    twoDIndexer = lambda x : (lambda f : (f(x[:,0]), f(x[:,1])))\
                            (lambda y : [] if not y.size else y)

    def toThreeChannel (im1) :
        """ Takes a single channel grayscale image and
        converts it to 3 channel grayscale"""

        im3 = np.empty (im1.shape + (3, ), dtype=np.uint8)
        for i in [0, 1, 2] :
            im3[:,:,i] = np.copy(im1)
        return im3

    def stripDict (dic, bands) :
        """ Takes a dictionary attribute of a galaxy object and
        strips it down to contain contain keys in argument 'bands'"""

        return dic if not bands else {b:_ for b,_ in dic.items() if b in bands}

    def copyRet (dic, bands, asDict) :
        """ Repeated condition in diagnosis methods """
        return dic if asDict or not bands or len(bands) > 1 else dic[bands]

    def __init__ (self, objid, cood, fitsFold, bands="ri") :
        """Constructor for the galaxy object
            objid       - Object id                 (from .csv file)
            cood        - Coordinates of the object (from .csv file)
            fitsFold    - Directory for FITS files  (Supplied by Batch object)
            bands       - Which bands for object    (Supplied by Batch object) """

        self.objid = objid
        self.bands = bands
        self.cood = cood
        self.fitsFold = fitsFold
        self.repoLink = None

        # Initialising dictionaries to None
        (self.downLinks,
        self.cutouts,           # Init to None to save I/O
        self.imgs,              # Never None
        self.hullInds,          # Never None
        self.hullRegs,          # Never None
        self.freqRegs,          # Never None (All 0s, in fact)
        self.freqSmooth,        # Never None (All 0s, in fact)
        self.noises,            # None
        self.peaks,             # Never None
        self.gtype) = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}

        ######################################################################
        # If download link fails, all else fails
        # If download link succeeds, 'cutouts', 'gtype' must compulsorily have a
        # value
        ######################################################################
        for b in bands :
            self.cutouts[b] = None
            self.gtype[b] = None

        batchlog.info ("{} --> Initialised".format(self.objid))

    def __str__ (self) :
        """Galaxy object to string"""
        return self.objid

    def getFitsPath (self, b) :
        """ Helper function for an object that returns the FITS file path
        for the band specified by argument 'b'"""

        return os.path.join (self.fitsFold, "{}-{}.fits".format(self.objid, b))

    def setRepoLink (self) :
        """Set the FITS repository link (for all bands)"""

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

    def setBandLinks (self) :
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
        """For a given band 'b', downloads the .bz2 container of the FITS file
        from the argument 'dlink' and extract it """

        # Download path for the .bz2
        dPath = os.path.join(self.fitsFold, dlink[dlink.rfind('/')+1:])

        try :
            # Downloading .bz2 to 'dPath'
            urllib.request.urlretrieve(dlink, dPath)
        except Exception as e :
            batchlog.error("{} --> Error in obtaining .bz2 for {} band. Moving onto next object".format(self.objid, b))
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
            batchlog.error("{} --> Error in extraction of .bz2 for {} band. Moving onto next object".format(self.objid, b))
            raise e

    def download (self) :
        """Downloads the frame for an object id
        for all bands as specified by Batch object"""

        # Bands that remain to be downloaded. Ignores invalid bands in 'band' attribute
        toDown = [b for b in self.bands if b in "ugriz" and not os.path.exists(self.getFitsPath(b))]
        batchlog.debug ("toDown = {}".format(toDown))
        if not toDown :
            batchlog.info ("{} --> FITS files of all bands already downloaded".format(self.objid))
            return
        batchlog.info ("{} --> FITS bands to be downloaded - {}".format(self.objid, toDown))

        # Initialised at constructor
        self.setRepoLink() if self.repoLink is None else None

        # Initialised at constructor
        if self.repoLink is None :
            # Do at the classification stage
            batchlog.info("{} --> Will set gtype to INVALID_OBJID".format(self.objid))
            return
        batchlog.info("{} --> FITS repository link successfully retrieved".format(self.objid))

        # Initialised at constructor
        self.setBandLinks() if not self.downLinks else None
        batchlog.info("{} --> FITS bands download links retrieved".format(self.objid))

        # Looping over bands
        for b, dlink in self.downLinks.items() :
            if b in self.bands and b in "ugriz" :
                # Download only if the FITS file doesn't exist
                if not os.path.exists (self.getFitsPath(b)) :
                    self.downloadExtract (b, dlink)
                batchlog.info("{} --> Obtained FITS file for {}-band".format(self.objid, b))

    def cutout (self, rad=40) :
        """Performs a cutout centred at attribute 'cood' for radius 'rad'
        for all bands"""

        def cutout_b (fitsPath, rad) :
            """ Helper function that performs cutout for a band
            for a given radius in argument 'rad'"""

            # In case the FITS file was not downloaded due to any error
            if not os.path.exists (fitsPath) :
                batchlog.warning ("{} --> No cutout performed as FITS file doesn't exist".format(self.objid))
                return None

            hdu = fits.open (fitsPath, memmap=False)[0]
            wcs = WCS(hdu.header)
            position = SkyCoord(ra=Angle (self.cood[0], unit=u.deg),
                            dec=Angle (self.cood[1], unit=u.deg))
            size = u.Quantity ((rad, rad), u.arcsec)
            return Cutout2D (hdu.data, position, size, wcs=wcs).data

        # Looping over bands
        for b in self.bands :
            ######################################################################
            # Considers valid bands only and does cutout only if dict[band] is
            # still None as I/O is costly
            ######################################################################
            if b in "ugriz" and self.cutouts[b] is None :
                self.cutouts[b] = cutout_b (self.getFitsPath(b), rad)
            batchlog.info("{} --> Got cutout for {}-band".format(self.objid, b))

    def smoothen (self, reduc=2, sgx=5, sgy=5) :
        """Performs a smoothening on the raw cutout data as follows (for each band) -
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

        def smoothen_b (img, reduc, sgx, sgy) :
            """Helper function for that performs smoothening for a given
            band. Arguments are described in enclosing method"""

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

        # Looping over bands
        for b, cut in self.cutouts.items() :
            self.imgs[b] = smoothen_b(cut, reduc, sgx, sgy)
            batchlog.info("{} --> Smoothened {}-band".format(self.objid, b))

    def hullRegion (self) :
        """ Finds the region where gradient ascent has to be performed for each band
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
                    entirely within the hull (Definition of convexity) """

        def hullRegion_b (img) :
            """Takes as input the smoothened image of a band and returns
            the hull boundary and the region contained between said boundary
            'img' is never None when this function is called"""

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

        # Looping over bands
        for b, img in self.imgs.items() :
            self.hullInds[b], self.hullRegs[b] = (Galaxy.emptyInds(), Galaxy.emptyInds()) if not img.size \
                                                else hullRegion_b(img)
            batchlog.info("{} --> Obtained hull boundary and region for {}-band".format(self.objid, b))

    def gradAsc (self, reduc=100, tol=3) :
        """Performs gradient ascent in the region indicated by attribute 'hullReg'
        for each band. It works as follows -
            1. Initialises a point randomly in the region
            2. In a 7x7 region, finds the pixel which has the
            highest value greater than the current pixel
                -> Note that 7x7 is to be interpreted as (2*tol + 1) x (2*tol + 1)
                at every piece of documentation in this method
                -> 'tol' defines the no. of neighbors to be considered during
                gradient ascent
            3. Sets the current pixel to this max pixel
            4. Repeat the above steps until no greater pixel can be found in the
            7x7 region

        The above routine is performed 'iter' times (defined in internal function
        gradAsc_b). The final peak (pixel coordinate) of each run of gradient ascent
        is compared with a running list of peaks and a filtration condition is applied
        to include it or not. This condition is described in detail in the comments

        'reduc' is a proportionality factor to decide the number of runs of gradient
        ascent to perform"""

        # Lambda function to checks if a 2-D point is in a list of points
        isPointIn = lambda pt, reg : (pt == reg).all(axis=1).any()

        ######################################################################
        # Returns the 7x7 neighborhood centred around a pixel
        # Any neighborhood point that falls outside the search region is
        # discarded
        ######################################################################
        tolNeighs = lambda pt, reg, t : [(pt[0]+dx, pt[1]+dy)
                                        for dx in range(-t,t+1) for dy in range(-t,t+1)
                                        if (dx != 0 or dy != 0)
                                        and isPointIn([pt[0]+dx, pt[1]+dy], reg)]

        def gradAsc_b (reg, gradKey, reduc, tol) :
            """ Performs gradient ascent on a region for a particular band
            Argument 'gradKey' is a lambda function for the underlying pixel
            values for the band image """

            batchlog.debug("Size of region = {}".format(len(reg)))
            if not reg.size :
                return Galaxy.emptyPeaks()

            ######################################################################
            # The runs of gradient ascent is taken to be proportional to the
            # length of the region
            ######################################################################
            iters = max(len(reg)//reduc, 100)

            # Running peak list. {(x,y) : [pixel value, frequency]}
            peaks = {}

            ######################################################################
            # Filtration condition -
            # 1. All the points in a 3x3 region around the peak must be in the
            # search region
            #   -> Otherwise the peak found is too close to the edge
            # 2. The peak found must not have an underlying pixel value of 0
            #   -> This occurs because the hull does enclose a lot of noise
            #   !!!! TO BE GENERALISED LATER !!!
            #
            # Each line in the lambda function below represents the conditions
            # in the above order
            ######################################################################
            peakFilter = lambda pk :\
            len(tolNeighs(pk, reg, 1)) == 8 and \
            gradKey(pk) != 0

            # No need to keep track of iteration number
            for _ in range(0, iters) :
                # Step 1
                st = pk = pt = (lambda x:(x[0], x[1]))(reg[np.random.choice(range(0, reg.shape[0]))])
                while pt :
                    ######################################################################
                    # If at any point in the interation, the list of neighboring 7x7 points
                    # is empty, it means that the point has gotten trapped somewhere. This
                    # can occur when the region under consideration is miniscule
                    ######################################################################
                    if not (ns:=tolNeighs(pt, reg, tol)) :
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

        # Looping over bands
        for b,img in self.imgs.items() :
            self.peaks[b] = Galaxy.emptyPeaks() if not img.size\
                            else gradAsc_b(self.hullRegs[b], lambda x:img[x], reduc, tol)
            batchlog.info("{} --> Found peak list for {}-band".format(self.objid, b))

    def setFreq (self) :
        """Computes the frequency of pixel values in the smooth
        and hull region for each band"""

        def calcFreq (series, freq) :
            """ Internal function that returns the frequency list in [0-255] """
            uq = np.unique(series, return_counts=True)
            ind, val = tuple(np.split(np.array([
                                            [i, uq[1][ag][0]]
                                            for i in range(0,256)
                                            if not not (ag:=np.argwhere(uq[0]==i).flatten()).size
                                            ]), 2, axis=1))
            freq[ind] = val
            return freq

        for b, series in self.getSmoothSeries().items() :
            freq = np.zeros(256, dtype=np.uint)
            self.freqSmooth[b] = freq if not series.size else calcFreq (series, freq)
            batchlog.info("{} --> Set the smooth frequency list for {}-band".format(self.objid, b))

        for b, series in self.getRegSeries().items() :
            freq = np.zeros(256, dtype=np.uint)
            self.freqRegs[b] = freq if not series.size else calcFreq (series, freq)
            batchlog.info("{} --> Set the hull region frequency list for {}-band".format(self.objid, b))

    def getSmooth (self, bands="", triple=False, asDict=False) :
        """ Returns a copy of the smoothed image of the specified band(s)
        If asDict is set,
            then images are returned as a dictionary
        Else,
            it's returned as a dictonary only if there are multiple bands
            otherwise,
                the image of the band is returned

        Additionally, if triple is set, then it is the three-channel copy of
        the image that is returned"""

        # If triple =

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

    def getRegSeries (self, bands="", asDict=False) :
        """ Flattens the hull region """

        regFlat = {b:Galaxy.emptySeries() if not img.size else img[Galaxy.twoDIndexer(self.hullRegs[b])].flatten()
                for b,img in
                Galaxy.stripDict(self.imgs, bands).items()
            }
        return Galaxy.copyRet(regFlat, bands, asDict)

    def getHullMarked (self, bands="", asDict=False) :
        """ Returns a copy of the smoothed image of the specified band(s) with hull marked"""

        hullMarks = self.getSmooth(bands, True, True)
        for b, hi in Galaxy.stripDict(self.hullInds, bands).items() :
            hullMarks[b][Galaxy.twoDIndexer(hi)] = np.array(Galaxy.hullMarker)

        return Galaxy.copyRet(hullMarks, bands, asDict)

    def getPeakMarked (self, bands="", hull=False, asDict=False) :
        """ Returns a copy of the smoothed image of the specified band(s) with peaks marked
        Additionally, if hull is set, then the hull is marked as well"""

        peakMarks = self.getHullMarked(bands, True) if hull else self.getSmooth(bands, True, True)
        for b,pks in Galaxy.stripDict(self.peaks, bands).items() :
            for pk in pks :
                peakMarks[b][pk] = np.array(Galaxy.peakMarker)

        return Galaxy.copyRet(peakMarks, bands, asDict)
