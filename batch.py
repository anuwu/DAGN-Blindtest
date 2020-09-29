import os
import sys
import numpy as np
import pandas as pd
import logging as log
import datetime as dt
import matplotlib.pyplot as plt
from importlib import reload
from textwrap import TextWrapper as txwr
from PIL import Image

import galaxy
galaxy = reload(galaxy)
Galaxy = galaxy.Galaxy

# Create the /Logs folder for the root directory if it doesn't already exist
if not os.path.isdir ("Logs") :
    os.mkdir ("Logs")

def dateFmt () :
    """Returns the date component of the run log file"""
    dtStr = str(dt.datetime.now())
    dtStr = dtStr[:dtStr.find('.')]
    dtStr = dtStr.replace(' ', '_')
    return dtStr

# Set the logger for this run of classifications
runlog = log.getLogger (__name__)
runlog.setLevel (log.INFO)
runLogPath = "Logs/run_{}.log".format(dateFmt())
fileHandler = log.FileHandler (runLogPath)
fileHandler.setFormatter (log.Formatter ("%(levelname)s : RUN_INIT : %(asctime)s : %(message)s",
                         datefmt='%m/%d/%Y %I:%M:%S %p'))

# Ensures that there is only one fileHandler for the current logger
for h in runlog.handlers :
    runlog.removeHandler (h)

runlog.addHandler (fileHandler)
runlog.info("Batch runner started!")

sys.setrecursionlimit(10**6)

class Batch () :
    """Class that defines a batch of SDSS objIDs on which classifcation is to be performed"""

    def getBatch (batchName, bands="ri", rad=40, csv=None) :
        """Class method to get a batch"""

        try :
            batch = Batch(batchName, (batchName + ".csv") if csv is None else csv,
                    bands, rad)
        except (FileNotFoundError, ValueError) as e :
            print ("Error initialising batch!")
            print("Kindly check the latest message in the logfile '{}' for a fix.".format(
                os.path.join(os.getcwd(), runLogPath)
            ))
            print ("Abort!")
            batch = None
        finally :
            return batch

    def logFixFmt (fix, k=50) :
        """Formats error messages for the run logger"""
        return  2*(k*"#" + '\n') + txwr(width=k).fill(text=fix) + '\n' + 2*(k*"#" + '\n')

    #############################################################################################################
    #############################################################################################################

    def __setBatchLogger__ (self) :
        """Sets the correct fileHandler location in the galaxy.py logger
        for the batch of galaxies that will be processsed. Also deletes the
        default logger in it if it exists"""

        fh = log.FileHandler(self.logPath)
        fh.setFormatter (log.Formatter ("%(levelname)s : GALAXY : %(asctime)s : %(message)s",
                                 datefmt='%m/%d/%Y %I:%M:%S %p'))

        # Ensuring only one fileHandler is associated with the batch logger
        for h in galaxy.batchlog.handlers :
            galaxy.batchlog.removeHandler (h)

        galaxy.batchlog.addHandler (fh)
        galaxy.batchlog.info ("Logger (re)set!")

    def __prelude__ (self) :
        """Sets up files and folders and checks for existence of
        folder indicated by attribute batchName and the 'csv' filename"""

        # To access fileHandler of the logger
        global fileHandler
        fileHandler.setFormatter (log.Formatter ("%(levelname)s : RUN_INIT : %(asctime)s : %(message)s",
                             datefmt='%m/%d/%Y %I:%M:%S %p'))

        # Checks if the /Data directory has been created at the root directory
        runlog.info("Checking environment for the new batch.")
        if not os.path.isdir("Data") :
            runlog.critical("Data folder not found!\n\n{}".format(Batch.logFixFmt(
                "Please create a folder named 'Data' in the notebook directory and rerun!"
            )))
            raise FileNotFoundError

        # Checks if batchName folder exists in /Data
        if not os.path.isdir (self.batchFold) :
            runlog.critical("Batch folder not found\n\n{}".format(Batch.logFixFmt(
                "Please create a folder for the batch at '{}' and rerun!".format(self.batchFold)
            )))
            raise FileNotFoundError

        ######################################################################
        # Checks if the .csv file exists. If the 'csv' argument is None, the
        # name of the .csv file is taken to be the same name as its containing
        # folder
        ######################################################################
        runlog.debug("{} and {}".format(self.csvName, self.csvPath))
        if not os.path.exists (self.csvPath) :
            runlog.critical("Batch .csv file at path '{}' not found\n\n{}".format(self.batchFold, Batch.logFixFmt(
                "Please supply the name of the appropriate .csv file and rerun!"
            )))
            raise FileNotFoundError

        # Changing name of the run log fileHandler to reflect the batch it is
        # presently handling
        runlog.info("Valid environment! Changing log format to handle batch '{}'".format(self.batchName))
        fileHandler.setFormatter (log.Formatter ("%(levelname)s : {} : %(asctime)s : %(message)s".format(self.batchName),
                         datefmt='%m/%d/%Y %I:%M:%S %p'))

        # Ensures only one fileHandler exists
        for h in runlog.handlers :
            runlog.removeHandler(h)
        runlog.addHandler(fileHandler)

        ######################################################################
        # Creates a /FITS folder in the batch folder where all the FITS files will
        # be stored
        ######################################################################
        if not os.path.exists (self.fitsFold) :
            os.mkdir (self.fitsFold)
            runlog.info("Created FITS folder for batch")
        else :
            runlog.info("FITS folder for the batch already exists")

    def __init__ (self, batchName, csvName, bands="ri", rad=40) :
        """Constructor for the batch. Fills galaxy dictionary"""

        self.batchName = batchName
        self.csvName = csvName
        self.__prelude__ ()
        runlog.info("Successfully created environment for batch")

        # Function to check if the band(s) supplied by the user is valid
        areBandsValid = lambda bs : len([b for b in bs if b in "ugriz"]) == len(bs) != 0
        runlog.debug (bands)
        runlog.debug (len(bands))

        ######################################################################
        # If the bands are not valid, a warning is logged
        # This is because the Galaxy object internally takes care of
        # invalid bands
        ######################################################################
        if not areBandsValid(bands) :
            runlog.warning("One or more bands in '{}' invalid\n\n{}".format(bands, Batch.logFixFmt(
            "Please ensure that bands are a combination of 'ugriz' only!"
            )))
            raise ValueError("Invalid Band. Please use 'ugriz'")

        self.bands = bands
        try :
            df = pd.read_csv(self.csvPath, dtype=object, usecols=["objID", "ra", "dec"])
        except ValueError as e :
            runlog.critical("Invalid columns in .csv file\n\n{}".format(Batch.logFixFmt(
                "Please ensure columns 'objID', 'ra' and 'dec' are present in the .csv \
                file (in that order) and rerun!"
                )))
            raise e

        ######################################################################
        # Sets the logger for the batch. This log file exists in the
        # batch Folder
        ######################################################################
        self.__setBatchLogger__ ()
        runlog.info("Set the batch logger")

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # The list comprehension below is where I can insert extra code to start
        # the classification midway
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        self.galaxies = [Galaxy(str(objid), (ra, dec), self.fitsFold, self.bands)
                        for objid, ra, dec in zip(df["objID"], df["ra"], df["dec"])]

        runlog.info ("Number of galaxies to process - {}".format(len(self.galaxies)))

    def __str__ (self) :
        """Batch object to string"""
        return self.csvPath

    def __len__ (self) :
        """Length of the batch"""
        return len(self.galaxies)

    @property
    def csvPath (self) :
        """Property attribute - Plain name of the csv File"""
        return os.path.join(os.getcwd(), self.batchFold, self.csvName)

    @property
    def batchFold (self) :
        """Property attribute - Path of the batch folder"""
        return os.path.join (os.getcwd(), "Data/", self.batchName)

    @property
    def fitsFold (self) :
        """Property attribute - Path of the FITS folder for the batch"""
        return os.path.join (os.getcwd(), "Data/", self.batchName, "FITS")

    @property
    def logPath (self) :
        """Property attribute - Path of the log file for the batch"""
        return os.path.join(os.getcwd(), self.batchFold, "{}.log".format(self.batchName))

    def downloadBatch (self) :
        """For each galaxy in the batch's list, downloads it
        to the FITS folder of the batch"""

        runlog.info("Downloading currently monitoring batch")
        for i, g in enumerate(self.galaxies) :
            try :
                g.download()
            except Exception as e :
                runlog.error("Unknown error while downloading! Please check exception message\n\n{}".format(Batch.logFixFmt(str(e))))
            else :
                runlog.info("{} --> Downloaded".format(g.objid))

        runlog.info("Downloaded currently monitoring batch")

    def loadBatch (self) :
        """Loads all the FITS files of the batch into memory"""

        runlog.info ("Loading currently monitoring batch")
        for g in self.galaxies :
            try :
                g.cutout ()
            except Exception as e :
                runlog.error("Unknown error in loading FITS! Please check exception message\n\n{}".format(Batch.logFixFmt(str(e))))
            else :
                runlog.info("{} --> Loaded".format(g.objid))

        runlog.info("Loaded currently monitoring batch")

    def processBatch (self) :
        """For each galaxy in the batch's list, processes it
        to the pre-classification stage. Performs the following -
            1. Cutout from the FITS file
            2. Smoothen raw cutout data
            3. Find the hull region where peak searching is done
            4. Perform peak searching """

        runlog.info ("Processing currently monitoring batch")
        for g in self.galaxies :
            g.smoothen()
            g.hullRegion()
            g.distInfo()
            g.filter()
            runlog.info("{} --> Processed".format(g.objid))

        runlog.info ("Processed currently monitoring batch")

    def classifyBatch (self) :
        """ For each galaxy in the batch, classifies it only if the
        filtration dict allows it to
        1. Identifies the noise/signal cutoff by fitting a gaussian to
        the inverse intensity histogram
        2. Performs stochastic gradient ascent  !!! TO-DO
        3. Applies conditions of connected regions to classify the peaks !!! TO-DO"""

        runlog.info ("Classifying currently monitoring batch")
        for g in self.galaxies :
            g.fitGaussian()
            g.searchRegion()
            g.sga()
            g.classify()
            runlog.info("{} --> Classified".format(g.objid))

        runlog.info ("Classified currently monitoring batch")

    #############################################################################################################
    #############################################################################################################

    def procDiagnose (self, constrictHist=False, invHist=False) :
        """Generates the following for each galaxy in each band -
            1. Peaks with enclosing hull
            2. Histogram of smoothed image
            3. Histogram of hull region"""

        diagPath = os.path.join(self.batchFold, "Proc-Diag")
        if not os.path.isdir(diagPath) :
            os.mkdir(diagPath)

        runlog.info ("Diagnosing currently monitoring batch")
        for g in self.galaxies :
            for b, filt in g.filtrate.items() :
                if filt :
                    continue

                # Hull only
                img = g.getHullMarked(b)
                svimg = Image.fromarray(img.astype(np.uint8))
                svimg.save(os.path.join(diagPath, "{}-{}_hull.png".format(g.objid, b)))

                # Hull with signal
                img = g.getHullMarked(b, True)
                svimg = Image.fromarray(img.astype(np.uint8))
                svimg.save(os.path.join(diagPath, "{}-{}_hullSignal.png".format(g.objid, b)))

                # Scatter and fit
                argPack = g.getGaussPlot(b)
                for i in [0, 1, 2] :
                    plt.plot(*argPack[i]["args"], **argPack[i]["kwargs"])
                plt.savefig (os.path.join(diagPath, "{}-{}_gaussFit.png".format(g.objid, b)))
                plt.close()

            runlog.info("{} --> Diagnosed".format(g.objid))

        runlog.info ("Diagnosed currently monitoring batch")
