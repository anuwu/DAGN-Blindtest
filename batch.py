import os
import pandas as pd
import logging as log
import datetime as dt
import galaxy
from textwrap import TextWrapper as txwr

if not os.path.isdir ("Logs") :
    os.mkdir ("Logs")

def dateFmt () :
    """Date component of the run log file"""
    dtStr = str(dt.datetime.now())
    dtStr = dtStr[:dtStr.find('.')]
    dtStr = dtStr.replace(' ', '_')
    return dtStr

Galaxy = galaxy.Galaxy
runlog = log.getLogger (__name__)
runlog.setLevel (log.INFO)
runLogPath = "Logs/run_{}.log".format(dateFmt())
fileHandler = log.FileHandler (runLogPath)
fileHandler.setFormatter (log.Formatter ("%(levelname)s : RUN_INIT : %(asctime)s : %(message)s",
                         datefmt='%m/%d/%Y %I:%M:%S %p'))
for h in runlog.handlers :
    runlog.removeHandler (h)

runlog.addHandler (fileHandler)
runlog.info("Batch runner started!")

def logFixFmt (fix, k=50) :
    """Formats error messages for the run logger"""
    return  2*(k*"#" + '\n') + txwr(width=k).fill(text=fix) + '\n' + 2*(k*"#" + '\n')

def runPrelude (batchName, csv) :
    """Sets up files and folders and checks for existence"""

    global fileHandler
    fileHandler.setFormatter (log.Formatter ("%(levelname)s : RUN_INIT : %(asctime)s : %(message)s",
                         datefmt='%m/%d/%Y %I:%M:%S %p'))
    '''
    if batchName not in Batch.logFHs :
        Batch.logFHs[batchName] = None
    '''

    runlog.info("Entering prelude to create new batch!")
    if not os.path.isdir("Data") :
        runlog.critical("Data folder not found!\n\n{}".format(logFixFmt(
            "Please create a folder named 'Data' in the notebook directory and rerun!"
        )))
        raise FileNotFoundError

    batchPath = "Data/{}".format(batchName)
    if not os.path.isdir (batchPath) :
        runlog.critical("Batch folder not found\n\n{}".format(logFixFmt(
            "Please create a folder for the batch at '{}' and rerun!".format(batchPath)
        )))
        raise FileNotFoundError

    csvName = batchName if csv is None else csv
    csvPath = batchPath + "/{}".format(csvName + ".csv")
    if not os.path.exists (csvPath) :
        runlog.critical("Batch .csv file at path '{}' not found\n\n{}".format(batchPath, logFixFmt(
            "Please supply the name of the appropriate .csv file and rerun!"
        )))
        raise FileNotFoundError

    runlog.info("Changing log format to handle batch '{}'".format(batchName))
    fileHandler.setFormatter (log.Formatter ("%(levelname)s : {} : %(asctime)s : %(message)s".format(batchName),
                     datefmt='%m/%d/%Y %I:%M:%S %p'))
    for h in runlog.handlers :
        runlog.removeHandler(h)

    runlog.addHandler(fileHandler)

    fitsPath = os.path.join (batchPath, "FITS")
    if not os.path.exists (fitsPath) :
        os.mkdir (fitsPath)
        runlog.info("Created FITS folder for batch")
    else :
        runlog.info("FITS folder for the batch already exists")

    runlog.info("Successfully created environment for batch\n\nMonitoring...")
    return csvPath

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

    def __init__ (self, batchName, bands='r', rad=40, csv=None) :
        """Constructor for the batch. Fills galaxy dictionary and gets the batch logger"""

        areBandsValid = lambda bs : len([b for b in bs if b in "ugriz"]) == len(bs) != 0
        runlog.debug (bands)
        runlog.debug (len(bands))
        if not areBandsValid(bands) :
            runlog.warning("One or more bands in '{}' invalid\n\n{}".format(bands, logFixFmt(
            "Please ensure that bands are a combination of 'ugriz' only!"
            )))
            raise ValueError("Invalid Band. Please use 'ugriz'")

        self.batchName = batchName
        self.csvPath = runPrelude (batchName, csv)
        try :
            df = pd.read_csv(self.csvPath, dtype=object, usecols=["objID", "ra", "dec"])
        except ValueError as e :
            runlog.critical("Invalid columns in .csv file\n\n{}".format(logFixFmt(
                "Please ensure columns 'objID', 'ra' and 'dec' are present in the .csv \
                file (in that order) and rerun!"
                )))
            raise e

        galaxy.setBatchLogger (batchName, self.batchPath)

        self.galaxies = {}
        for objid, ra, dec in zip(df["objID"], df["ra"], df["dec"]) :
            objid = str(objid)
            self.galaxies[objid] = Galaxy(objid, (ra, dec), self.fitsPath)

    @property
    def csvName (self) :
        """Plain name of the csv File"""
        return (lambda st:st[1+st.rfind('/'):st.find(".csv")])(self.csvPath)

    @property
    def batchPath (self) :
        return os.path.join ("Data/", self.batchName)

    @property
    def fitsPath (self) :
        return os.path.join ("Data/", self.batchName, "FITS")

    def downloadPhase (self) :
        runlog.info("Downloading currently monitoring batch")
        for objid, g in self.galaxies.items() :
            try :
                g.download()
            except Exception as e :
                runlog.error("Unknown error! Please check exception message\n\n{}".format(logFixFmt(str(e))))
            else :
                runlog.info("{} --> Downloaded".format(objid))
        runlog.info("Downloaded currently monitoring batch")

    def procPhase (self) :
        runlog.info ("Processing currently monitoring batch")
        for objid, g in self.galaxies.items() :
            g.cutout()
            g.smooth()
            g.hullRegion()
            g.gradAsc()
        runlog.info ("Processed currently monitoring batch")
