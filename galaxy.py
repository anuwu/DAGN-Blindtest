import os
import requests
import urllib
import bs4
import bz2
import warnings
import numpy as np
import cv2
import matplotlib.colors as colors
import logging as log

from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord, Angle, Latitude, Longitude
from astropy import units as u
from astropy.utils.exceptions import AstropyWarning
from collections import OrderedDict
from enum import Enum

warnings.simplefilter('ignore', category=AstropyWarning)

batchlog = log.getLogger (__name__)
batchlog.setLevel(log.DEBUG)
fileHandler = log.FileHandler("./galaxy.log")
fileHandler.setFormatter (log.Formatter ("%(levelname)s : GALAXY : %(asctime)s : %(message)s",
                         datefmt='%m/%d/%Y %I:%M:%S %p'))
if not batchlog.handlers :
    batchlog.addHandler(fileHandler)

def setBatchLogger (batchName, batchPath) :
    fh = log.FileHandler(os.path.join(batchPath, "{}.log".format(batchName)))
    fh.setFormatter (log.Formatter ("%(levelname)s : {} : %(asctime)s : %(message)s".format(batchName),
                             datefmt='%m/%d/%Y %I:%M:%S %p'))

    for h in batchlog.handlers :
        batchlog.removeHandler (h)
    batchlog.addHandler (fh)


class Galaxy () :
    """Contains all of the galaxies info"""
    class GalType (Enum) :
        (INVALID_OBJID,
         NO_NOISE,
         NO_CENTRE,
         NO_PEAK,
         SINGLE,
         DOUBLE) = tuple(range(6))

    cannyLow = 25
    cannyHigh = 50
    borderColor = (0, 0, 255)

    def __init__ (self, objid, cood, fitsFold, bands='r') :
        """Simple constructor. Other attributes populated with methods"""
        self.objid = objid
        self.bands = bands
        self.cood = cood
        self.repoLink = None
        self.fitsFold = fitsFold

        # If download link fails, all else fails
        # If download link succeeds, cutouts, gtype must compulsorily have a value
        (self.downLinks, self.cutouts, self.noise,
         self.imgs, self.hullReg, self.peaks, self.gtype) = {}, {}, {}, {}, {}, {}, {}

        for b in bands :
            self.cutouts[b] = None
            self.gtype[b] = None

        batchlog.info ("Initialised galaxy object {}".format(objid))

    def getFitsPath (self, b) :
        return os.path.join (self.fitsFold, "{}-{}.fits".format(self.objid, b))

    def download (self) :
        """Downloads the frame for an object id"""

        def getRepoLink (objid) :
            link = "http://skyserver.sdss.org/dr14/en/tools/explore/summary.aspx?objid=" + objid
            soup = bs4.BeautifulSoup(requests.get(link).text, features='lxml')

            # Condition for invalid link
            if len(soup.select(".nodatafound")) == 1 :
                return None

            tagType = type(
                bs4.BeautifulSoup('<b class="boldest">Extremely bold</b>' , features = 'lxml').b
            )

            for c in soup.select('.s') :
                tag = c.contents[0]
                if tagType == type(tag) and (fitsLinkTag:=str(tag)).find('Get FITS') > -1 :
                    break

            fitsLinkTag = fitsLinkTag[fitsLinkTag.find('"')+1:]
            fitsLinkTag = fitsLinkTag[:fitsLinkTag.find('"')].replace("amp;",'')
            return "http://skyserver.sdss.org/dr15/en/tools/explore/" + fitsLinkTag

        def getBandLinks (repoLink) :
            def procClass (st) :
                st = st[st.find('href='):]
                st = st[st.find('"')+1:]
                return st[:st.find('"')]

            return {(lambda s: s[s.rfind('/') + 7])(procClass(str(x))) : procClass(str(x))
            for x in
            bs4.BeautifulSoup(requests.get(repoLink).text, features = 'lxml').select(".l")[0:5]
            }

        def downloadExtract (b, dlink, fitsFold, fitsPath) :
            dPath = os.path.join(fitsFold, dlink[dlink.rfind('/')+1:])
            urllib.request.urlretrieve(dlink, dPath)
            zipf = bz2.BZ2File(dPath)
            data = zipf.read()
            zipf.close()
            extractPath = dPath[:-4]
            open(extractPath, 'wb').write(data) ;
            os.rename(extractPath, fitsPath)
            os.remove(dPath)

        allDown = True
        for b in self.bands :
            if not os.path.exists (self.getFitsPath(b)) :
                allDown = False

        if allDown :
            return
        if self.repoLink is None :
            self.repoLink = getRepoLink(self.objid)
            if self.repoLink is None :
                self.gtype = {b : GalType.INVALID_OBJID for b in self.bands}
                return
        if not self.downLinks :
            self.downLinks = getBandLinks(self.repoLink)

        for b, dlink in self.downLinks.items() :
            if b in self.bands and not os.path.exists (p:=self.getFitsPath(b)) :
                downloadExtract (b, dlink, self.fitsFold, p)

    def cutout (self, rad=40) :
        def cutout_b (fitsPath) :
            nonlocal rad

            hdu = fits.open (fitsPath, memmap=False)[0]
            wcs = WCS(hdu.header)
            position = SkyCoord(ra=Angle (self.cood[0], unit=u.deg),
                                dec=Angle (self.cood[1], unit=u.deg))
            size = u.Quantity ((rad, rad), u.arcsec)
            return Cutout2D (hdu.data, position, size, wcs=wcs).data


        for b in self.bands :
            if self.cutouts[b] is None :
                self.cutouts[b] = cutout_b (self.getFitsPath(b))


    def smooth (self, reduc=2, sgx=5, sgy=5) :
        def smooth_b (img) :
            nonlocal reduc, sgx, sgy

            img[img < 0] = 0
            img[img >= 0] -= np.min(img[img >= 0])

            if (vmin:=max(0.1, np.median(img)/reduc)) >= (vmax:=np.max(img)) :
                return None

            imgNorm = colors.LogNorm(vmin, vmax, True).__call__(img)
            imgNorm.data[imgNorm.mask] = 0

            return (lambda d:cv2.GaussianBlur(np.floor(255*(lambda x,mn,mx : (x-mn)/(mx-mn))
                                                           (d, np.min(d), np.max(d))
                                                    ).astype(np.uint8), (sgx,sgy),0
                    ))(imgNorm.data)

        for b, cut in self.cutouts.items() :
            self.imgs[b] = smooth_b(np.copy(cut))

    def hullRegion (self) :
        get_canny_edges = lambda s,l,h : np.argwhere(cv2.Canny(s, l, h) == 255)
        def toThreeChannel (im1) :
            im3 = np.empty (im1.shape + (3, ), dtype=np.uint8)
            for i in [0, 1, 2] :
                im3[:,:,i] = im1
            return im3

        def hullRegion_b (img) :
            fullImg = cv2.drawContours (toThreeChannel(np.copy(img)),
                                    [cv2.convexHull(np.flip(get_canny_edges(img,
                                                                            Galaxy.cannyLow,
                                                                            Galaxy.cannyHigh),
                                                            axis=1))],
                                       0, Galaxy.borderColor, 1, 8)

            return None if fullImg is None\
            else (lambda f, uq : np.array([
                [f[uq[i],0],y]
                for i in range(0, len(uq))
                    for y in (lambda ys:np.setdiff1d(range(np.min(ys), np.max(ys)+1), ys))\
                            (f[uq[i]:,1]
                            if i == len(uq)-1
                            else f[uq[i]:uq[i+1],1])
                                        ])
            )(*(lambda f : (f, np.unique(f[:,0], True)[1]))\
                (np.argwhere((fullImg == np.array(Galaxy.borderColor)).all(axis=2))))


        for b, img in self.imgs.items() :
            self.hullReg[b] = None if img is None else hullRegion_b(img)

    def gradAsc (self, reduc=100, tol=3) :
        isPointIn = lambda pt, reg : (pt == reg).all(axis=1).any()
        tolNeighs = lambda pt, reg, t : [(pt[0]+dx, pt[1]+dy)
                                           for dx in range(-t,t+1) for dy in range(-t,t+1)
                                            if (dx != 0 or dy != 0)
                                           and isPointIn([pt[0]+dx, pt[1]+dy], reg)]
        def gradAsc_b (reg, gradKey) :
            nonlocal reduc, tol

            iters = max(len(reg)//reduc, 100)
            peaks = {}
            peakFilter = lambda pk : np.array([
                x not in peaks for x in tolNeighs(pk, reg, tol)
            ]).all() and len(tolNeighs(pk, reg, 1)) == 8 and gradKey(pk) != 0

            for _ in range(0, iters) :
                st = pk = pt = (lambda x:(x[0], x[1]))(reg[np.random.choice(range(0, reg.shape[0]))])
                while pt :
                    ns = tolNeighs (pt, reg, tol)
                    if not ns :
                        break

                    mxn = max(ns, key = gradKey)
                    pt = mxn if gradKey(mxn) > gradKey(pk:=pt) else None

                if pk not in peaks :
                    if peakFilter (pk) :
                        peaks[pk] = [gradKey(pk), 1]
                else :
                    peaks[pk][1] += 1

            return OrderedDict (sorted (peaks.items(), key=lambda x:x[1][0], reverse=True))

        for b in self.bands :
            self.peaks[b] = None
            if (img:=self.imgs[b]) is not None and (reg:=self.hullReg[b]) is not None :
                self.peaks[b] = gradAsc_b(self.hullReg[b], lambda x:img[x])

    def markPeaks (self) :
        for b, peaks in self.peaks.items() :
            peaks = self.peaks[b]
