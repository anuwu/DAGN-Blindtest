import os
import sys
import bs4
import warnings
import numpy as np
import logging

from importlib import reload

import sdss_scrape as scrap
import plane_coods as pc
import light_profile as lp
import fits_proc as fp
import peaks as pk

# Setting the logger
log = logging.getLogger(__name__)
log.setLevel(logging.WARNING)
fileHandler = logging.FileHandler("./galaxy.log", mode='w')
fileHandler.setFormatter(logging.Formatter("%(levelname)s : GALAXY : %(asctime)s : %(message)s",
                         datefmt='%m/%d/%Y %I:%M:%S %p'))

# Ensures only one file handler
for h in log.handlers :
    log.removeHandler(h)

log.addHandler(fileHandler)
log.info("Welcome!")

class Galaxy () :
    """
    Contains FITS image data of an SDSS galaxy
    and associated data extracted out of this image
    to detect whether it is a double nuclei galaxy or not
    """

    # For marking the hulls
    hullMarker = (0, 0, 255)
    # Color for marking the peaks found by shc
    pointMarker = (255, 0, 0)
    # Color for marking signal region in hull
    signalMarker = (64, 224, 208)

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

    def __init__ (self, objid, cood, fitsFold, bands="ugriz") :
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
        ) = objid, bands, cood, fitsFold, None

        ######################################################################
        # Initialising dictionaries to None. The convention used is that if
        # one of the dict attributes is None or Empty (check class lambda funcs)
        # then, the dict attributes following it are None, Empty be default

        # This chain is followed until the 'filtrate' attribute. This is because,
        # all the computations beyond that point are very heavy. They'll be done
        # only for bands in which there's a reasonable guarantee of finding
        # a single/double galaxy
        ######################################################################
        (self.downLinks,        # 1. FITS download links in all bands           --> Empty Dict or Full
        self.cutouts,           # 2. Raw cutout data                            --> None signifies FITS Failure
        self.imgs,              # 3. Smoothened cutout image                    --> Empty (0, 2) numpy array signifies no appreciable signal in image / FITS failure

        ######################## Failures cascade down #######################

        self.hullCoods,         # 4. Convex hull indices                        --> Empty (0, 2) numpy array signifies no appreciable signal in image
        self.hullRegs,          # 5. Region enclosed by convex hull             --> Empty (0, 2) numpy array signifies tightly enclosed hull

        ########################### Failure WALL ############################

        self.filtrate,          # 6. Condition not to process any further       --> Finding an object/signal is highly improbable

        ######################## Failures cascade down #######################

        self.dists,             # 7. (mean, sigma, noise, half-width half-max)  --> LightProfile() object
        self.peaks              # 8. Reduced peak list after dfs                --> Peak() object
        ) = {}, {}, {}, {}, {}, {}, {}, {}

        ######################################################################
        # Cutouts is init to None to signify -
        # 1. FITS file has failed to download/extract
        # 2. FITS file has failed to open
        # 3. Cutout from FITS file has failed
        ######################################################################
        self.cutouts = {b:None for b in bands if b in "ugriz"}

        log.info("{} --> Initialised".format(self.objid))

    def __str__ (self) :
        """ Galaxy object to string """
        return self.objid

    def download (self) :
        """
        Downloads the frame for an object id for all bands.
        1. If repository link is None, then the object ID is invalid
        2. If the download links dict is empty, then scraping the links has failed
        """

        # Bands that remain to be downloaded. Ignores invalid bands in 'band' attribute
        toDown = [b for b in self.bands if b in "ugriz" and not os.path.exists(self.getFitsPath(b))]
        if not toDown :
            log.info("{} --> FITS files of all bands already downloaded".format(self.objid))
            return
        log.info("{} --> FITS bands to be downloaded - {}".format(self.objid, toDown))

        # Initialised at constructor
        try :
            self.repoLink = scrap.scrapeRepoLink(self.objid)
        except scrap.RepoScrapeError as e :
            self.peaks = {b:pk.Peak(b, None, pk.GalType.DOWN_FAIL) for b in self.bands if b in "ugriz"}
            log.warning("{} --> Setting all peaks as DOWN_FAIL : {}".format(self.objid, e.msg))

        if self.repoLink is None :
            self.peaks = {b:pk.Peak(b, pk.GalType.INVALID_OBJID) for b in self.bands if b in "ugriz"}
            log.warning("{} --> Setting all peaks as INVALID_OBJID".format(self.objid))
            return

        log.info("{} --> FITS repository link retrieved".format(self.objid))

        # Initialised at constructor
        try :
            self.downLinks = scrap.scrapeBandLinks(self.repoLink)
        except scrap.BandScrapeError as e :
            self.peaks = {b:pk.Peak(b, None, pk.GalType.DOWN_FAIL) for b in self.bands if b in "ugriz"}
            log.warning("{} --> Setting all peaks as DOWN_FAIL : {}".format(self.objid, e.msg))

        log.info("{} --> FITS bands download links retrieved".format(self.objid))

        for b, dlink in self.downLinks.items() :
            if b in self.bands :
                # Download only if the FITS file doesn't exist
                if not os.path.exists(self.getFitsPath(b)) :
                    try :
                        scrap.downloadExtract(self.objid, b, dlink, self.fitsFold, self.getFitsPath(b))
                    except scrap.BZ2DownError as e :
                        log.warning("{} --> {}-band : {}".format(self.objid, b, e.msg))
                    except scrap.BZ2ExtractError as e :
                        log.warning("{} --> {}-band : {}".format(self.objid, b, e.msg))

                log.info("{} --> Obtained FITS file for {}-band".format(self.objid, b))

    def cutout (self, rad=40) :
        """ Gets the cutout of images in all bands. Check fits_proc.py """

        for b in self.bands :
            ######################################################################
            # Considers valid bands only and does cutout only if cutout of that band
            # hasn't already been loaded in memory
            ######################################################################
            if b in "ugriz" and self.cutouts[b] is None :
                self.cutouts[b] = fp.cutout(self.getFitsPath(b), self.cood, rad)
                log.info("{} --> Got cutout for {}-band".format(self.objid, b))

    def smoothen (self, reduc=2, sgx=5, sgy=5) :
        """ Smoothens the image in all bands """

        for b, cut in self.cutouts.items() :
            self.imgs[b] = fp.smoothen(cut, reduc, sgx, sgy)
            log.info("{} --> Smoothened {}-band".format(self.objid, b))

    def hullRegion (self, low=25, high=50) :
        """
        Finds a region via edge detection and convex hull where the
        peak search will be carried out. Check fits_proc.py
        """

        for b, img in self.imgs.items() :
            self.hullCoods[b], self.hullRegs[b] = fp.hullRegion(img, low, high, Galaxy.hullMarker)
            log.info("{} --> Obtained hull boundary and region for {}-band".format(self.objid, b))

    def filter (self) :
        """
        Filtration condition before classification -
            0. If peak has already been set to invalid
            1. Cutout is None                                       --> FITS failure
            2. Smoothed image is empty                              --> No appreciable signal that can be processed well by LogNorm
            3. Hull boundary is empty                               --> No sharp object in the image, mostly dark
            4. Hull region is empty                                 --> Object too small to conduct shc
            5. Centre is not in hull region                         --> Hull encloses an object with a different SDSS objID
        If true, do not classify. Finding an object/signal is highly improbable
        """

        # This will occur only if all bands are set to INVALID_OBJID or DOWN_FAIL
        if not self.peaks :
            for b, cuts in self.cutouts.items() :
                self.filtrate[b] = cuts is None or\
                                not self.imgs[b].size or\
                                not self.hullCoods[b].size or\
                                not self.hullRegs[b].size or\
                                not pc.isPointIn((self.imgs[b].shape[0]//2, self.imgs[b].shape[1]//2), self.hullRegs[b])
                log.info("{} --> Calculated filtrate for {}-band".format(self.objid, b))
        else :
            self.filtrate = {b:True for b in self.bands if b in "ugriz"}

    def fitProfile (self, profile='sersic') :
        """
        Receives a distribution object for each band based on the
        light distribution in the hull region
        """

        lightProfs = {
        'gaussian'      : lp.Gaussian,
        'sersic'        : lp.Sersic
        }

        for b, series in self.getSmoothRegSeries().items() :
            if self.filtrate[b] :
                self.dists[b] = None
                log.info("{} --> Light profile fitting for {}-band filtered out".format(self.objid, b))
            else :
                if profile == 'sersic' :
                    sersFail = False
                    self.dists[b] = lightProfs[profile](series)
                    self.dists[b].fit()
                    if not (lp.Sersic.llim <= self.dists[b].params[1] <= lp.Sersic.rlim) :
                        sersFail = True
                        log.info("Failed to fit Sersic profile. Reverting to Gaussian")

                if profile == 'gaussian' or (profile == 'sersic' and sersFail) :
                    self.dists[b] = lightProfs['gaussian'](series)
                    self.dists[b].fit()

                self.dists[b].inferLevels()
                self.dists[b].estimateNoise(self.cutouts[b], self.imgs[b])
                log.info("{} --> Fit the light profile, inferred noise/signal and SNR noise for {}-band".format(self.objid, b))

    def setPeaks (self) :
        """ Sets the Peak object for each band """

        for b, filt in self.filtrate.items() :
            if filt :
                self.peaks[b] = pk.Peak(b, None, pk.GalType.FILTERED)
                log.info("{} --> Peak setting for {}-band filtered out".format(self.objid, b))
            else :
                self.peaks[b] = pk.Peak(b, lambda p:self.imgs[b][p])
                self.peaks[b].setRegion(self.hullRegs[b], self.dists[b].noise)
                self.peaks[b].shc(lambda p : self.cutouts[b][p], self.dists[b].noiseSNR)
                self.peaks[b].filterPeaks(tuple(np.array(self.imgs[b].shape)//2),
                                        self.dists[b].signal)
                log.info("{} --> Set peaks for {}-band".format(self.objid, b))

            self.peaks[b].setType()

    def csvLine (self) :
        """
        Returns the line which will be simply written to the .csv
        file corresponding to this galaxy. Galaxy type along with position
        of peaks in order of 'ugriz'
        """

        band_entry = {b:pk.csvColumn() for b, pk in self.peaks.items()}
        args = tuple([band_entry[b] for b in "ugriz"])
        return "{},{},{},{},{}".format(*args)

    def progressLine (self) :
        """ Returns a line to output the batch's progress """

        st = ""
        for _, p in self.peaks.items() :
            st += str(p.btype)
            st += 2*' '

        return st

    #############################################################################################################
    #############################################################################################################

    def getFitsPath (self, b) :
        """
        Helper function for an object that returns the FITS file path
        for the band specified by argument 'b'
        """

        return os.path.join(self.fitsFold, "{}-{}.fits".format(self.objid, b))

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

        cuts = {b:fp.emptyImage if cut is None else cut
            for b,cut
            in Galaxy.stripDict(self.cutouts, bands).items()}
        return Galaxy.copyRet(cuts, bands, asDict)

    def getCutoutSeries (self, bands="", asDict=False) :
        """ Flattens the cutout image """

        cutFlat = {b:cut.flatten()
                for b, cut
                in self.getCutout(bands, True).items()}
        return Galaxy.copyRet(cutFlat, bands, asDict)

    def getCutoutRegSeries (self, bands="", asDict=False) :
        """ Flattens the hull region for cutout image """

        cutRegFlat = {b:cut[pc.coodIndexer(self.hullRegs[b])].flatten()
                    for b, cut
                    in self.getCutout(bands, True).items()}
        return Galaxy.copyRet(cutRegFlat, bands, asDict)

    def getSmooth (self, bands="", triple=False, asDict=False) :
        """ Returns the smoothed image """

        imgs = {b:fp.toThreeChannel(np.zeros_like(self.cutouts[b]) if not img.size else img) if triple
            else np.copy(img)
            for b, img
            in Galaxy.stripDict(self.imgs, bands).items()}

        return Galaxy.copyRet(imgs, bands, asDict)

    def getSmoothSeries (self, bands="", asDict=False) :
        """ Flattens the smoothed image """

        imgFlat = {b:img.flatten()
                for b, img
                in self.getSmooth(bands, False, True).items()}

        return Galaxy.copyRet(imgFlat, bands, asDict)

    def getSmoothRegSeries (self, bands="", asDict=False) :
        """ Flattens the hull region for smoothed image"""

        smoothRegFlat = {b:img[pc.coodIndexer(self.hullRegs[b])].flatten()
                        for b,img
                        in self.getSmooth(bands, False, True).items()}

        return Galaxy.copyRet(smoothRegFlat, bands, asDict)

    def getHullMarked (self, bands="", sig=False, asDict=False) :
        """ Returns a copy of the smoothed image of the specified band(s) with hull marked"""

        hullMarks = self.getSmooth(bands, True, True)
        for b, hi in Galaxy.stripDict(self.hullCoods, bands).items() :
            img = hullMarks[b]
            img[pc.coodIndexer(hi)] = np.array(Galaxy.hullMarker)
            if sig :
                noise = 0 if self.filtrate[b] else self.dists[b].noise
                img[img[...,0] > noise] = np.array(Galaxy.signalMarker)

        return Galaxy.copyRet(hullMarks, bands, asDict)

    def getPointsMarked (imgs, pointDict) :
        """ Helper function to mark the peaks """

        for b, pts in pointDict.items() :
            for pt in pts :
                imgs[b][pt] = np.array(Galaxy.pointMarker)

    def getHillOptsMarked (self, bands="", hull=False, asDict=False) :
        """
        Returns a copy of the smoothed image of the specified band(s)
        with raw peaks from shc, marked.
        Additionally, if hull is set, then the hull is marked as well
        """

        imgs = self.getHullMarked(bands, False, True) if hull else self.getSmooth(bands, True, True)
        Galaxy.getPointsMarked(imgs, {b:list(self.peaks[b].hillOpts) for b in bands})
        return Galaxy.copyRet(imgs, bands, asDict)

    def getPeaksMarked (self, bands="", hull=False, asDict=False) :
        """
        Returns a copy of the smoothed image of the specified band(s)
        with classified peaks marked
        Additionally, if hull is set, then the hull is marked as well
        """

        imgs = self.getHullMarked(bands, False, True) if hull else self.getSmooth(bands, True, True)
        Galaxy.getPointsMarked(imgs, {b:self.peaks[b].filtPeaks for b in bands})
        return Galaxy.copyRet(imgs, bands, asDict)

    def getProfilePlot (self, bands="", asDict=False) :
        """ Returns 4 sets of plotting arguments for the intensity distribution """

        argPacks = {b: lp.LightProfile.emptyArgs if filt else self.dists[b].plotArgs()
                    for b, filt
                    in self.filtrate.items()}

        return Galaxy.copyRet(argPacks, bands, asDict)
