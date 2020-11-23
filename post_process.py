import requests
import bs4
from astropy.wcs.utils import pixel_to_skycoord

import peaks
import plane_coods as pc
GalType = peaks.GalType

def isPairPure(dl1, dl2) :
    """
            Peak 1 in list 1 <- Neighs of peak 1 of list 2
            and
            Peak 2 in list 1 <- Neighs of peak 2 of list 2
        or
            Peak 1 in list 1 <- Neighs of peak 2 of list 2
            and
            Peak 2 in list 1 <- Neighs of peak 1 of list 2
    """

    return (dl1[0] in pc.purityNeighs(dl2[0]) and dl1[1] in pc.purityNeighs(dl2[1])) \
    or \
    (dl1[0] in pc.purityNeighs(dl2[1]) and dl1[1] in pc.purityNeighs(dl2[0]))

def isPure (dlist) :
    """
    Takes in a list of double peaklists, and determines
    their purity. Purity is defined as whether all the peak
    coordinates lie in consistently the same location (with a
    neighborhood tolerance) across all bands
    """

    if len(dlist) == 1 :
        return True

    purity = []
    for i, dl1 in enumerate(dlist) :
        pure = []
        for j, dl2 in enumerate(dlist) :
            if i != j : pure.append(isPairPure(dl1, dl2))

        purity.append(False not in pure)

    return False not in purity

def get_purity_band (gal) :
    """
    Takes in a galaxy object and does the following -
        1. If there is a double returns if it's pure
        and a representative band of purity.
        2. If there is no double, returns None
    """

    plist = [p.filtPeaks
        for b, p in gal.peaks.items()
        if p.btype == GalType.DOUBLE
    ]

    if not plist :
        return None, None

    band = [b for b, p in gal.peaks.items() if p.btype == GalType.DOUBLE][0]
    return isPure(plist), band

def get_bands_csv (gal, bands) :
    """ Returns the band column of the pure csv """

    return "".join([
        b for b in bands
        if gal.peaks[b].btype == GalType.DOUBLE
    ])

def cood_to_objid (cood) :
	"""
	Takes (ra, dec) coordinates and tries to return an SDSS objid
	if an object is catalogued at that coordinate
	"""

	link = "https://skyserver.sdss.org/dr12/en/tools/explore/summary.aspx?ra={}&dec={}"\
			.format(cood.ra.deg, cood.dec.deg)
	soup = bs4.BeautifulSoup(requests.get(link).text, features='lxml')
	if len(soup.select(".nodatafound")) == 1 :
		return None

	st = str(soup.findAll("td", {"class": "t"})[6])
	return st[st[:-1].rfind('>')+1 : st.rfind('<')]

def peak_to_objid (wcs, plist) :
	""" Takes an SDSS objid, path to the FITS file
	and returns the SDSS objids (if they exist) corresponding
	to the pixel coordinates of the peaks in 'plist' """

	cood1 = pixel_to_skycoord(plist[0][0], plist[0][1], wcs)
	cood2 = pixel_to_skycoord(plist[1][0], plist[1][1], wcs)
	return cood_to_objid(cood1), cood_to_objid(cood2)
