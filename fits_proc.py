import os
import cv2
import warnings
import logging
import matplotlib.colors as colors
import numpy as np

from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord, Angle
from astropy import units as u
from astropy.utils.exceptions import AstropyWarning

# Setting the logger
log = logging.getLogger(__name__)
log.setLevel(logging.WARNING)
fileHandler = logging.FileHandler("./fits_proc.log", mode='w')
fileHandler.setFormatter(logging.Formatter("%(levelname)s : FITS_PROC : %(asctime)s : %(message)s",
                         datefmt='%m/%d/%Y %I:%M:%S %p'))

# Ensures only one file handler
for h in log.handlers :
    log.removeHandler(h)

log.addHandler(fileHandler)
log.info("Welcome!")

# Removes the 'RADECSYS deprecated' warning from astropy cutout
warnings.simplefilter('ignore', category=AstropyWarning)

# Empty (0,) numpy array
emptyArray = np.array([])
# Empty (0,2) numpy array
emptyArray2D = np.array([]).reshape(-1, 2)
# Empty 2-D image
emptyImage = emptyArray2D
# Empty list of indices
emptyCoods = emptyArray2D

def toThreeChannel (im1) :
    """ Takes a single channel grayscale image and converts it to 3 channel grayscale """

    im3 = np.empty (im1.shape + (3, ), dtype=np.uint8)
    for i in [0, 1, 2] :
        im3[:,:,i] = np.copy(im1)
    return im3

def cutout (fitsPath, cood, rad) :
    """
    Performs cutout of FITS file -
        fitsPath            - Directory where FITS file is present
        cood                - (ra, dec) of the object
        rad                 - Radius (in arcseconds) of the cutout
    """

    # In case the FITS file was not downloaded due to any error
    if not os.path.exists(fitsPath) :
        log.warning("No cutout performed as FITS file doesn't exist at '{}'".format(fitsPath))
        return None

    try :
        hdu = fits.open(fitsPath, memmap=False)[0]
    except Exception as e :
        log.warning("Error in loading FITS file at {}. Returning None".format(fitsPath))
        return None

    log.info("Opened FITS file at '{}'".format(fitsPath))
    wcs = WCS(hdu.header)
    position = SkyCoord(ra=Angle(cood[0], unit=u.deg),
                    dec=Angle(cood[1], unit=u.deg))
    size = u.Quantity((rad, rad), u.arcsec)

    log.info("Performed cutout for '{}'".format(fitsPath))
    return Cutout2D(hdu.data, position, size, wcs=wcs)

def smoothen (img, reduc, sgx, sgy) :
    """
    Performs a smoothening on the raw cutout data as follows (for each band) -
        img         - Image on which the smoothening will be performed
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

    if img is None :
        log.info("Image supplied for smoothening was empty. Returning")
        return emptyImage

    # Step 1
    img[img < 0] = 0

    # Step 2
    img[img >= 0] -= np.min(img[img >= 0])

    # Step 3
    vmin = max(0.1, np.median(img)/reduc)
    vmax = np.max(img)
    if vmin >= vmax :
        log.warning("Appreciable intensity not found while smoothing")
        return emptyImage

    # Step 4
    imgNorm = colors.LogNorm(vmin, vmax, True).__call__(img)
    log.info("LogNorm of image done with (vmin, vmax) = {}".format((vmin, vmax)))

    # Step 5
    imgNorm.data[imgNorm.mask] = 0

    # Step 6
    mn = np.min(imgNorm.data)
    mx = np.max(imgNorm.data)
    gray = np.floor(255*(imgNorm.data - mn)/(mx - mn)).astype(np.uint8)

    # Step 7
    smoothed = cv2.GaussianBlur(gray, (sgx, sgy), 0)

    log.info("Image was smoothed")
    return smoothed

def hullRegion (img, low, high, hullMarker) :
    """
    Finds the convex hull of all edges output by canny edge detector -
        1. Uses canny with parameters 'low', 'high' on the argument 'img'
        to find edges
        2. Takes the convex hull of these edges to find the boundary
            a. cv2.convexHull actually returns the set of minimal points
            needed to define the contour
            b. cv2.drawContours uses this to draw the actual boundary on
            an image
            c. The boundary is then found by searching for the boundary
            color marker (hullMarker as an rgb 3-tuple)
        3. Scans the boundary from left to right to find the region
        it contains
            -> This works due to the property of the hull being convex
            -> For any two points in the region contained in the hull,
            there is a straight line that connects them which exists
            entirely within the hull (Definition of convexity)
    """

    if not img.size :
        log.info("Image supplied for edge detection was empty. Returning")
        return emptyCoods, emptyCoods

    # Step 1
    # Finding edges returned by Canny
    cannyEdges = np.argwhere(cv2.Canny(img, low, high) == 255)

    # There are no well defined edges if 'cannyEdges' is empty
    if not cannyEdges.size :
        log.info("Not edges found by Canny. Returning")
        return emptyCoods, emptyCoods

    # Step 2a
    cannyHull = [cv2.convexHull(np.flip(cannyEdges, axis=1))]
    # Step 2b
    fullImg = cv2.drawContours(toThreeChannel(img), cannyHull, 0, hullMarker, 1, 8)
    # Step 2c
    hullCoods = np.argwhere((fullImg == np.array(hullMarker)).all(axis=2))

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
        (hullCoods))

    log.info("Found hull boundary and regions of lengths {} and {}, respectively".\
            format(len(hullCoods), len(hullRegs)))
    return hullCoods, emptyCoods if not hullRegs.size else hullRegs
