import bs4
import requests
import urllib
import bz2
import os
from astropy import units as u
from astropy.coordinates import SkyCoord

def is_valid_obj_link (soup) :
	############################################################
	# Returns True or False
	#
	# Sometimes it occurs that a valid object coordinates result
	# in an invalid object page. In such a case, the page will have
	# only one '.nodatafound' class and that is selected
	############################################################



    class_nodatafound = soup.select(".nodatafound")
    if (len(class_nodatafound) == 1) :
        return False 
    else :
        return True

def obj_link_FITS_repository_link_sexa_cood (obj_link) :
	############################################################
	# Returns the FITS repository link and the scraped sexagesimal
	# coordinates. 
	#
	# The coordinates in the object page and that from
	# the resolver are different. The former usually has more decimal
	# points and they do not differ much.
	#
	# The FITS repository link contains download links to the FITS
	# files of various bands in U , V , R , I , Z. 
	############################################################



    res = requests.get(obj_link)		# Downloads the object page
    soup = bs4.BeautifulSoup(res.text , features = 'lxml')		# Soupifying the page
    class_s = soup.select('.s')			# The link to the FITS repository lies in a '.s' class
    
    flag = is_valid_obj_link (soup)		# Checking for validity of the object page
    if not flag :
        return (None,None) 			# This return value is used in if-else checking in the driver code.
    
    # Creating a litmus tag-type for comparison ...
    soupy = bs4.BeautifulSoup('<b class="boldest">Extremely bold</b>' , features = 'lxml')
    tag = soupy.b
    tag_type = type(tag)
    
    for i in class_s :
        if (tag_type == type(i.contents[0]) and str(i.contents[0]).find('Get FITS') > -1) :
            fits_link_tag = str(i.contents[0])
            break
   
   	# Procressing the HTML tag containing the FITS link to result in a link to the FITS repository.
    fits_link_tag = fits_link_tag[fits_link_tag.find('"')+1:]
    fits_link_tag = fits_link_tag[:fits_link_tag.find('"')]
    fits_link_tag = fits_link_tag.replace("amp;",'')
    fits_repository_link = "http://skyserver.sdss.org/dr15/en/tools/explore/" + fits_link_tag

    #########################################################################################
    #########################################################################################

    class_large = soup.select(".large")		# The sexagesimal coordinates in the object page lie in '.large' class.
    for i in class_large :
        sexa_cood_tag = str(i)
        break 
        

    # Processing the HTML tag containing the sexagesimal coordinates ...
    sexa_cood_tag = sexa_cood_tag[sexa_cood_tag.find('">')+2:]
    sexa_cood = sexa_cood_tag[:sexa_cood_tag.find('<')]    
    sexa_cood = sexa_cood.replace(',' , '')
    sexa_cood = list(sexa_cood)
    sexa_cood[sexa_cood.index(':')] = 'h'
    sexa_cood[sexa_cood.index(':')] = 'm'
    sexa_cood[sexa_cood.index(':')] = 'd'
    sexa_cood[sexa_cood.index(':')] = 'm'
    sexa_cood = ''.join(sexa_cood)
    sexa_cood = sexa_cood.replace(' ' , 's ')
    sexa_cood = sexa_cood + 's'

   
    return (fits_repository_link , sexa_cood)

def FITS_repository_rband_download_link (fits_repository_link) :
	############################################################
	# Returns a download link to the R-band file
	############################################################



    res = requests.get(fits_repository_link)		# Downloads the FITS repository page
    soup = bs4.BeautifulSoup(res.text , features = 'lxml')		# Soupifying the page

    class_content = soup.select(".l")		# The download link to the R-band FITS file is in a '.l' class
    for i in class_content :
        if (str(i).find('frame-r') > -1) :
            download_link_tag = str(i)
            break

    # Processing the HTML tag containing the download link ..
    download_link_tag = download_link_tag[download_link_tag.find('href='):]
    download_link_tag = download_link_tag[download_link_tag.find('"')+1:]
    download_link = download_link_tag[:download_link_tag.find('"')]

    return download_link


def download_extract (download_link , obj_name, csv_abs_path) :
	############################################################
	# Doesn't return anything. Downloads the compressed FITS file
	# and then extracts, renames it.
	############################################################



    download_filename = download_link[download_link.rfind('/') + 1:]		# Default download filename
    download_filepath = csv_abs_path + "/Data/" + download_filename		# In /Data/
    obj_name_path = csv_abs_path + "/Data/" + obj_name +".fits"		# Renamed filename
    
    if os.path.exists(obj_name_path) :		# If renamed file already exists, then the process is already complete.
        return 
    
    if os.path.exists(download_filepath) :		# I don't think this is needed, but code works fine.
        return

    urllib.request.urlretrieve(download_link , download_filepath)   # Downloading the zip file.
    
    zipfile = bz2.BZ2File(download_filepath) # open the file
    data = zipfile.read() # get the decompressed data
    extracted_path = download_filepath[:-4] # assuming the filepath ends with .bz2
    open(extracted_path, 'wb').write(data) ;
    
    try :
        os.rename(extracted_path , csv_abs_path + "/Data/" + obj_name + ".fits")		# Renaming default name to correct object name.
    except FileExistsError :
        pass
        
    zipfile.close ()
    os.remove(download_filepath)		# Removing the downloaded zip file.
    if os.path.exists(extracted_path) :
        os.remove(extracted_path)