import os
import pandas as pd
import logging as log
import datetime as dt
import galaxy
from textwrap import TextWrapper as txwr

# Create the /Logs folder for the root directory if it doesn't already exist
if not os.path.isdir ("Logs") :
    os.mkdir ("Logs")

def dateFmt () :
    """Returnsthe date component of the run log file"""
    dtStr = str(dt.datetime.now())
    dtStr = dtStr[:dtStr.find('.')]
    dtStr = dtStr.replace(' ', '_')
    return dtStr

# Renaming for easy creation of Galaxy objects
Galaxy = galaxy.Galaxy

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

def logFixFmt (fix, k=50) :
    """Formats error messages for the run logger"""
    return  2*(k*"#" + '\n') + txwr(width=k).fill(text=fix) + '\n' + 2*(k*"#" + '\n')

class Batch () :
    """Class that defines a batch of SDSS objIDs on which classifcation is to be performed"""

    def getBatch (batchName, bands='r', rad=40, csv=None) :
        """Class method to get a batch"""

        try :
            batch = Batch (batchName, bands, rad, batchName if csv is None else csv)
        except (FileNotFoundError, ValueError) as e :
            print ("Error initialising batch!")
            print("Kindly check the latest message in the logfile '{}' for a fix.".format(
                os.path.join(os.getcwd(), runLogPath)
            ))
            print ("Abort!")
            batch = None
        finally :
            return batch

    def runPrelude (batchName, csv) :
        """Sets up files and folders and checks for existence of
        folder indicated by 'batchName' and the 'csv' filename"""

        # To access fileHandler of the logger
        global fileHandler
        fileHandler.setFormatter (log.Formatter ("%(levelname)s : RUN_INIT : %(asctime)s : %(message)s",
                             datefmt='%m/%d/%Y %I:%M:%S %p'))

        # Checks if the /Data directory has been created at the root directory
        runlog.info("Entering prelude to create new batch!")
        if not os.path.isdir("Data") :
            runlog.critical("Data folder not found!\n\n{}".format(logFixFmt(
                "Please create a folder named 'Data' in the notebook directory and rerun!"
            )))
            raise FileNotFoundError

        # Checks if batchName folder exists in /Data
        batchPath = "Data/{}".format(batchName)
        if not os.path.isdir (batchPath) :
            runlog.critical("Batch folder not found\n\n{}".format(logFixFmt(
                "Please create a folder for the batch at '{}' and rerun!".format(batchPath)
            )))
            raise FileNotFoundError

        ######################################################################
        # Checks if the .csv file exists. If the 'csv' argument is None, the
        # name of the .csv file is taken to be the same name as its containing
        # folder
        ######################################################################
        csvName = batchName if csv is None else csv
        csvPath = batchPath + "/{}".format(csvName + ".csv")
        if not os.path.exists (csvPath) :
            runlog.critical("Batch .csv file at path '{}' not found\n\n{}".format(batchPath, logFixFmt(
                "Please supply the name of the appropriate .csv file and rerun!"
            )))
            raise FileNotFoundError

        # Changing name of the run log fileHandler to reflect the batch it is
        # presently handling
        runlog.info("Changing log format to handle batch '{}'".format(batchName))
        fileHandler.setFormatter (log.Formatter ("%(levelname)s : {} : %(asctime)s : %(message)s".format(batchName),
                         datefmt='%m/%d/%Y %I:%M:%S %p'))

        # Ensures only one fileHandler exists
        for h in runlog.handlers :
            runlog.removeHandler(h)

        runlog.addHandler(fileHandler)

        ######################################################################
        # Creates a /FITS folder in the batch folder where all the FITS files will
        # be stored
        ######################################################################
        fitsPath = os.path.join (batchPath, "FITS")
        if not os.path.exists (fitsPath) :
            os.mkdir (fitsPath)
            runlog.info("Created FITS folder for batch")
        else :
            runlog.info("FITS folder for the batch already exists")

        runlog.info("Successfully created environment for batch\n\nMonitoring...")
        return csvPath

    def __init__ (self, batchName, bands='r', rad=40, csv=None) :
        """Constructor for the batch. Fills galaxy dictionary"""

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
            runlog.warning("One or more bands in '{}' invalid\n\n{}".format(bands, logFixFmt(
            "Please ensure that bands are a combination of 'ugriz' only!"
            )))
            raise ValueError("Invalid Band. Please use 'ugriz'")

        self.batchName = batchName
        self.bands = bands
        self.csvPath = Batch.runPrelude (batchName, csv)
        try :
            df = pd.read_csv(self.csvPath, dtype=object, usecols=["objID", "ra", "dec"])
        except ValueError as e :
            runlog.critical("Invalid columns in .csv file\n\n{}".format(logFixFmt(
                "Please ensure columns 'objID', 'ra' and 'dec' are present in the .csv \
                file (in that order) and rerun!"
                )))
            raise e

        ######################################################################
        # Sets the logger for the batch. This log file exists in the
        # batch Folder
        ######################################################################
        galaxy.setBatchLogger (batchName, self.batchPath)

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # The list comprehension below is where I can insert extra code to start
        # the classification midway
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        self.galaxies = [Galaxy(str(objid), (ra, dec), self.fitsPath, self.bands)
                        for objid, ra, dec in zip(df["objID"], df["ra"], df["dec"])]

    @property
    def csvName (self) :
        """Attribute - Plain name of the csv File"""
        return (lambda st:st[1+st.rfind('/'):st.find(".csv")])(self.csvPath)

    @property
    def batchPath (self) :
        """Attribute - Path of the batch folder"""
        return os.path.join ("Data/", self.batchName)

    @property
    def fitsPath (self) :
        """Attribute - Path of the FITS folder for the batch"""
        return os.path.join ("Data/", self.batchName, "FITS")

    def downloadPhase (self) :
        """For each galaxy in the batch's list, downloads it
        to the FITS folder of the batch"""

        runlog.info("Downloading currently monitoring batch")
        # Looping through galaxy objects
        for g in self.galaxies :
            try :
                g.download()
            except Exception as e :
                runlog.error("Unknown error! Please check exception message\n\n{}".format(logFixFmt(str(e))))
            else :
                runlog.info("{} --> Downloaded".format(g.objid))
        runlog.info("Downloaded currently monitoring batch")

    def procPhase (self) :
        """For each galaxy in the batch's list, processes it
        to the pre-classification stage. Performs the following -
            1. Cutout from the FITS file
            2. Smoothen raw cutout data
            3. Find the hull region where peak searching is done
            4. Perform peak searching """

        runlog.info ("Processing currently monitoring batch")
        # Looping through galaxy objects
        for g in self.galaxies :
            g.cutout()
            g.smooth()
            g.hullRegion()
            g.gradAsc()
            runlog.info("{} --> Processed".format(g.objid))
        runlog.info ("Processed currently monitoring batch")
