import os
import requests
import urllib
import bz2
import bs4
import logging

# Setting the logger
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
fileHandler = logging.FileHandler("./sdss_scrape.log", mode='w')
fileHandler.setFormatter(logging.Formatter("%(levelname)s : SDSS_SCRAPE : %(asctime)s : %(message)s",
                         datefmt='%m/%d/%Y %I:%M:%S %p'))

# Ensures only one file handler
for h in log.handlers :
    log.removeHandler(h)

log.addHandler(fileHandler)
log.info("Welcome!")

######################################################################
# The following exception classes are defined so that the galaxy object
# can inform the peak object to set the appropriate result enum
######################################################################

class RepoScrapeError (Exception) :
    """ Exception class for failure on scraping repository link """

    def __init__ (self, msg) :
        """ Initialises the message of the exception """
        self.msg = msg

class BandScrapeError (Exception) :
    """ Exception class for failure on scraping download links of bands """

    def __init__ (self, msg) :
        """ Initialises the message of the exception """
        self.msg = msg

class BZ2DownError (Exception) :
    """ Exception class for the failure of downloading bz2 arhive """

    def __init__ (self, msg) :
        """ Initialises the message of the exception """
        self.msg = msg

class BZ2ExtractError (Exception) :
    """ Exception class for the failure of extraction of bz2 archive """

    def __init__ (self, msg) :
        """ Initialises the message of the exception """
        self.msg = msg

def scrapeRepoLink (objid) :
    """ Downloads the FITS repository link for a given SDSS objid """

    link = "http://skyserver.sdss.org/dr14/en/tools/explore/summary.aspx?objid=" + objid
    try :
        soup = bs4.BeautifulSoup(requests.get(link).text, features='lxml')
    except Exception as e :
        log.error("Failed to scrape repository link for {}".format(objid))
        raise RepoScrapeError("Failed to scrape repository link due to {}".format(type(e)))

    log.info("Request for {} skyserver summary successful".format(objid))

    # Condition for invalid link
    if len(soup.select(".nodatafound")) == 1 :
        log.warning("Objid {} is invalid".format(objid))
        return None

    # Finding the tag which contains the FITS repository link
    tagType = type(bs4.BeautifulSoup('<b class="boldest">Extremely bold</b>' , features = 'lxml').b)
    fitsLinkTag = None
    for c in soup.select('.s') :
        tag = c.contents[0]
        fitsLinkTag = str(tag)
        if tagType == type(tag) and fitsLinkTag.find('Get FITS') > -1 :
            break

    if fitsLinkTag is None :
        return None

    fitsLinkTag = fitsLinkTag[fitsLinkTag.find('"')+1:]
    fitsLinkTag = fitsLinkTag[:fitsLinkTag.find('"')].replace("amp;",'')

    repoLink = "http://skyserver.sdss.org/dr15/en/tools/explore/" + fitsLinkTag
    log.info("Scraped repository link for {} as '{}'".format(objid, repoLink))
    return repoLink

def procClass (st) :
    """Helper function to extract download link"""
    st = st[st.find('href='):]
    st = st[st.find('"')+1:]
    return st[:st.find('"')]

def scrapeBandLinks (repoLink) :
    """
    Computes a list of download links for the .bz2 archives (which contains)
    the FITS file, for each band, from the repository link
    Type of list --> [(Band, download link for band)]
    """

    try :
        dlinks = {(lambda s: s[s.rfind('/') + 7])(procClass(str(x))) : procClass(str(x))
        for x in
        bs4.BeautifulSoup(requests.get(repoLink).text, features = 'lxml').select(".l")[:5]
        }
    except Exception as e :
        log.error("Failed to scrape band links from '{}'".format(repoLink))
        raise BandScrapeError("Failed to scrape band links due to {}".format(type(e)))

    log.info("Scraped band links from '{}'".format(repoLink))
    return dlinks

def downloadExtract (objid, band, dlink, folder, fitsPath) :
    """
    Downloads the .bz2 archive for a FITS file and extracts it
        objid           - SDSS object
        band            - The band of the archive
        dlink           - Download link of the band
        folder          - Folder where the download/extraction will be done
        fname           - Final name of the FITS file
    """

    # Download path for the .bz2
    dPath = os.path.join(folder, dlink[dlink.rfind('/')+1:])

    try :
        urllib.request.urlretrieve(dlink, dPath)
    except Exception as e :
        log.error("Failed to download .bz2 archive for {} in {}-band".format(objid, band))
        raise BZ2DownError("Failed to download .bz2 archive due to {}".format(type(e)))

    log.info("Downloaded .bz2 for {} in {}-band to '{}'".format(objid, band, dPath))

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
        os.rename(extractPath, fitsPath)
        os.remove(dPath)
    except Exception as e :
        log.error("Failed to extract .bz2 archive for {} in {}-band".format(objid, band))
        raise BZ2ExtractError("Failed to extract .bz2 archive due to {}".format(type(e)))

    log.info("Extracted .bz2 for {} in {}-band to '{}'".format(objid, band, fitsPath))
