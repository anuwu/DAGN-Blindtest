import sys
import logging
import numpy as np
import plane_coods as pc
import fits_proc as fp

from enum import Enum
from collections import OrderedDict

# Override system recursion limit for DFS
sys.setrecursionlimit(10**8)

# Setting the logger
log = logging.getLogger(__name__)
log.setLevel(logging.WARNING)
fileHandler = logging.FileHandler("./peaks.log", mode='w')
fileHandler.setFormatter(logging.Formatter("%(levelname)s : PEAKS : %(asctime)s : %(message)s",
                         datefmt='%m/%d/%Y %I:%M:%S %p'))

# Ensures only one file handler
for h in log.handlers :
    log.removeHandler(h)

log.addHandler(fileHandler)
log.info("Welcome!")

def comparatorKey (cmp) :
    """Convert a cmp= function into a key= function"""
    class K:
        def __init__(self, key, *args):
            self.key = key
        def __lt__(self, other):
            return cmp(self.key, other.key)
        def __gt__(self, other):
            return not self.__lt__(other) and not self.__eq__(other)
        def __eq__(self, other):
            return not self.__lt__(other) and not cmp(other.key, self.key)
        def __le__(self, other):
            return self.__lt__(other) or self.__eq__(other)
        def __ge__(self, other):
            return not self.__lt__(other)
        def __ne__(self, other):
            return not self.__eq__(other)
    return K

class GalType (Enum) :
    """
    Enum for the final verdict of a galaxy
        INVALID_OBJID       - Self explanatory
        DOWN_FAIL           - Failed to download
        FILTERED            - Rejected by filtration condition
        NO_PEAK             - shc did not return any peaks
        SINGLE              - Single nuclei galaxy
        DOUBLE              - Double nuclei galaxy
    """

    (INVALID_OBJID,
    DOWN_FAIL,
    FILTERED,
    NO_PEAK,
    SINGLE,
    DOUBLE) = tuple(range(6))

    def __str__(self) :
        """ Enum as string """
        return self.name

class Peak () :
    """
    Stores data and methods to compute the intensity optima in a region of
    an image along with procedures to filter them to infer the true peaks
    """

    # Empty list of peaks
    emptyPeaks = OrderedDict({})

    def __init__ (self, band, hillKey=None, btype=None) :
        """Constructor for the Peak object
            band            - Band of the image
            hillKey         - 2D coordinate -> underlying pixel value for hill climbing
            btype           - Enum for the type of galaxy in this band
                            Usually called upon to set the peak state as -
                                -> INVALID_OBJID
                                -> DOWN_FAIL
                                -> FILTERED
        """

        self.band = band                                    # Band of the image for which peaks are found
        self.hillKey = hillKey                              # Underlying grayscale values for hill climbing
        self.reg = fp.emptyCoods                            # Region of hill climbing. Serves as input to hillKey
        self.hillOpts = Peak.emptyPeaks                     # Init hill climbing optima to the empty OrderedDict
        self.filtPeaks = []                                 # Init final peaks to the empty list
        self.comps = []                                     # Init connected components to the empty list
        self.btype = btype                                  # Init to None or an enum signifying failure as supplied by the caller
        log.info("Initialised peak object")

    def __str__ (self) :
        """ Peak object to string """
        return "{} : {}".format(self.btype, self.filtPeaks)

    def setType (self) :
        """
        Sets the enum after peaks has been found
            No peaks        - NO_PEAK
            One peak        - SINGLE
            Two peaks       - DOUBLE
        """

        if not self.filtPeaks :
            self.btype = GalType(GalType.NO_PEAK)
        elif len(self.filtPeaks) == 1 :
            self.btype = GalType(GalType.SINGLE)
        else :
            self.btype = GalType(GalType.DOUBLE)

        log.info("Set the type as {} after filtering peaks".format(self.btype))

    def setRegion (self, hullRegion, noise) :
        """ Sets the search region where stochastic hill climbing will be done"""

        if not noise :
            self.reg = self.hullRegion
            return

        self.reg = np.array([
            pt for pt in hullRegion if self.hillKey(tuple(pt)) >= noise
        ])
        log.info("Set the search region for SHC with length {}".format(len(self.reg)))

    def shc (self, snrKey, snrNoise, highPeak=False, signal=None, boundPeak=True, reduc=100, tol=3) :
        """
        Performs stochastic hill climbing -
        Data arguments -
            snrKey              - Underlying cutout pixel values for finding SNR
            snrNoise            - Average noise level for SNR calculation

        Parameter arguments -
            highPeak            - Whether or not to report peaks high enough in intensity
            signal              - Grayscale level above which peak must be detected when highPeak is True
            boundPeak           - Whether or not to report peaks near the edge of the region
            reduc               - Proportionality factor for the number of times to run SHC
            tol                 - Pixel resolution at which two peaks cannot be distinguished

        How it works -
            1. Initialises a point randomly in the region
            2. In a 7x7 region, finds the pixel which has the
            highest value greater than the current pixel
                -> Note that 7x7 is to be interpreted as (2*tol + 1) x (2*tol + 1)
                at every piece of documentation in this method
                -> 'tol' defines the no. of neighbors to be considered during
                shc
            3. Sets the current pixel to this max pixel
            4. Repeat the above steps until no greater pixel can be found in the
            7x7 region
        """

        if not self.reg.size :
            self.hillOpts = Peak.emptyPeaks()
            log.info("Region is empty for SHC. Returning")
            return

        ######################################################################
        # The runs of shc is taken to be proportional to the
        # length of the region
        ######################################################################
        iters = max(len(self.reg)//reduc, 100)

        # Running peak list. {(x,y) : [pixel value, frequency]}
        peaks = {}

        ######################################################################
        # Filtration condition -
        # 1. If boundPeak is true, All the points in a 1x1 region around the
        # peak must be in the search region
        #   -> Otherwise the peak found is too close to the edge
        # 2. If highPeak is true, the peak found must be above the half-width
        # half-maximum level
        # 3. The SNR must be greater than 3
        #   -> Found by averaging the pixels near the peak and dividing by the
        #   noise level
        # 4. No other neighbor in the 7x7 neighborhood of the peak must already
        # be in the peak list
        #
        # Each line in the lambda function below represents the conditions
        # in the above order
        ######################################################################
        validPeak = lambda pk :\
        (boundPeak or len(pc.neighsInReg(pk, self.reg, 1)) == 8) and\
        (not highPeak or self.hillKey(pk) >= signal) and\
        (not snrNoise or np.mean([snrKey(p) for p
                                in ([pk] + pc.neighsInReg(pk, self.reg, 1))
                                ])/snrNoise > 3) and\
        True not in [p in peaks
                    for p in pc.tolNeighs(pk, tol)]

        # No need to keep track of iteration number
        log.info("Performing SHC for {} iterations".format(iters))
        for _ in range(iters) :
            # Step 1
            st = pk = pt = tuple(self.reg[np.random.choice(range(len(self.reg)))])
            while pt :
                ######################################################################
                # If at any point in the interation, the list of neighboring 7x7 points
                # is empty, it means that the point has gotten trapped somewhere. This
                # can occur when the region under consideration is miniscule
                ######################################################################
                ns = pc.neighsInReg(pt, self.reg, tol)
                if not ns :
                    break

                # Step 2
                mxn = max(ns, key=self.hillKey)

                # Step 3
                pk = pt
                pt = mxn if self.hillKey(mxn) > self.hillKey(pt) else None # Step 4 (Termination)

            ######################################################################
            # If peak already exists, increase its frequency count
            # Otherwise, if the peak is allowed, create a new entry in the dict
            ######################################################################
            if pk in peaks :
                peaks[pk][1] += 1
            elif validPeak(pk) :
                peaks[pk] = [self.hillKey(pk), 1]

        self.hillOpts = OrderedDict(sorted(peaks.items(), key=lambda p:p[1][0], reverse=True))
        log.info("Performed SHC and found {} peaks".format(len(self.hillOpts)))

    def connectedComponents (self) :
        """ Returns the connected components in the region """

        def dfs (ind, reg, vis, comp) :
            """ Visiting function for depth first search """

            # Mark current index as visited
            vis[ind] = True
            # Append current index to the present component that is being created
            comp.append(ind)
            # For each neighbor, if it hasn't already been visited, visit it
            for i in pc.dfsNeighs(ind, reg) :
                if not vis[i]:
                    dfs(i, reg, vis, comp)

        noPts = len(self.reg)
        vis = np.repeat(False, noPts)
        comps = []
        seed = 0

        # Seed indices that each returns one connected component
        while True :
            comp = []
            dfs(seed, self.reg, vis, comp)
            # Mapping indices to mixel coordinates
            comps.append(np.array([
                self.reg[i] for i in comp
            ]))
            # Find the next seed index, if it exists
            seed = np.argmin(np.where(vis, 1, range(-noPts, 0, 1)))
            if not seed :
                break

        # Store the components
        self.comps = comps
        log.info("Found {} connected components".format(len(self.comps)))

    def filterPeaks (self, imCent, signal, distGrade=10) :
        """
        Filters the optimas returned by stochastic hill climbing
            imCent          - Coordinates of the centre of the image
            signal          - Level above which to select peaks

        How it works -
            1. If the no. of peaks returned by gradient descent is 0 or 1,
        then return it simply
            2. If the number of peaks is two or more, then we need to perform dfs
        and find the connected components. Following this stage, there are two
        variants towards classification -
                a. filterPeaks1 - Finds the top two brightest peaks and returns DOUBLE
                if they belong to the same component, else single
                b. filterPeaks2 - The connected regions are sorted according to a criteria
                    -> Smallest distance from the centre of the image (in bins of 10)
                    -> Largest size of a region (within each max)
            Top 1 or 2 peaks in this component is reported
        """

        # Return simply for 0 or 1 peak
        if not self.hillOpts :
            self.filtPeaks = []
            log.info("There is no hill optima. Returning")
            return
        if len(self.hillOpts) == 1 :
            self.filtPeaks = list(self.hillOpts)
            log.info("There is only one hill optima. Returning")
            return

        # Obtain connected components
        log.info("2 or more peaks. Running DFS")
        self.connectedComponents()

        ######################################################################
        # Returns the distance from the centre of the image to the centre of
        # the connected component
        ######################################################################
        compCentreDist = lambda cm,cent : np.floor(np.sqrt(np.sum(np.square(
            np.mean(cm, axis=0) - np.array(cent)//2
        ))))//distGrade

        # Dict --> component index : list of peaks
        comp_pk = {i:[p for p
                    in self.hillOpts
                    if pc.isPointIn(p, c)
                    ]
                for i,c in enumerate(self.comps)
        }
        # Dict --> component index : distance from centre of image to centre of component
        comp_dist = {i:compCentreDist(c, imCent)
                for i,c
                in enumerate(self.comps)
        }

        ######################################################################
        # Comparator function for regions -
        # 1. Ascending distance to the centre
        # 2. Descending size of region
        ######################################################################
        regLess = lambda c1, c2 : comp_dist[c1] < comp_dist[c2] \
                or (comp_dist[c1] == comp_dist[c2] and len(self.comps[c1]) > len(self.comps[c2]))

        # Sort the list of indices according to the comparator 'regless'
        compInds = sorted(range(len(self.comps)), key=comparatorKey(regLess))

        ######################################################################
        # Return the top two bright peaks in the best component after filtering
        # the ones, which are less than half-width half-maximum, out
        ######################################################################
        bestPeaks = [pk for pk
                in comp_pk[compInds[0]]
                if not signal or self.hillKey(pk) >= signal
        ]

        self.filtPeaks = sorted(bestPeaks, key=self.hillKey, reverse=True)[:2]
        log.info("Filtered down the peaks from SHC to {} peaks".format(len(self.filtPeaks)))

    def csvColumn (self) :
        """ Returns the entry for the result .csv column """

        pString = "["
        for p in self.filtPeaks :
            pString += str(p)
        pString += "]"

        return "{},\"{}\"".format(self.btype, pString)
