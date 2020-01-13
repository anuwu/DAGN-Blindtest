from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker, cm
import matplotlib.colors as colors
import cv2
import os
import sys
import importlib

def cutout_test () :
	print ("Initialisation complete!")

def cutout_fits (filename , sexa_cood, radius, csv_abs_path) :
	############################################################
	# Inputs - name of FITS file
	#		 - coordinates of the centre of object,
	#        - cutout radius
	#
	# Returns the FITS data as a 2-d array
	#
	# $hdul opens FITS file as an object
	# The first object of the FITS file contains the raw data
	# $wcs contains the world coordinate system alignment
	# $radius is the cutout radius
	############################################################

	filepath = csv_abs_path + "/Data/" + filename + ".fits" 

	hdul = fits.open(filepath , memmap = False)
	hdu = hdul[0]		
	wcs = WCS(hdu.header)
	position = SkyCoord(sexa_cood , frame = 'icrs')
	size = u.Quantity((radius,radius), u.arcsec)
	cutout = Cutout2D(hdu.data, position, size, wcs=wcs)

	hdu.data = cutout.data 		# Replacing original data with cutout-only data
	hdu.header.update(cutout.wcs.to_header())
	temppath = csv_abs_path + "/Data/" +filename+"_cut.fits" 	# New filename for object
	hdu.writeto(temppath , overwrite=True)		# Overwriting data
	hdul.close()		# Closing file handler to initiate removal of original FITS file
	del hdul[0].data 	# Ensuring no complications in file removal

	os.remove (filepath)

def cutout_fits2 (filename, csv_abs_path) :
	############################################################
	# Inputs - name of FITS file
	#		 - coordinates of the centre of object,
	#        - cutout radius
	#
	# Returns the FITS data as a 2-d array
	#
	# $hdul opens FITS file as an object
	# The first object of the FITS file contains the raw data
	# $wcs contains the world coordinate system alignment
	# $radius is the cutout radius
	############################################################

	filepath = csv_abs_path + "/Data/" + filename + ".fits" 

	hdul = fits.open(filepath , memmap = False)
	hdu = hdul[0]		
    
	return hdu.data


def export_png (pathname , X , Y , Z , r , c, csv_abs_path) :
	############################################################
	# Inputs - path to save the png file
	# 		 - X axis of grid
	#		 - Y axis of grid
	#        - Z values of the grid
	#        - rows of the png file
	#        - columns of the png file
	#        - dpi of the screen
	#        - offset of the screen
	#        - whether FITS file contains negative values or not

	# Exports a png file of size $r x $c
	############################################################

	fig = plt.gcf () 
	DPI = fig.get_dpi ()
	fig.set_size_inches ((r)/float(DPI) , (c)/float(DPI))

	ax = fig.add_subplot(111)
	ax.axis('off')
	ax.margins(0,0)
	ax.xaxis.set_major_locator(plt.NullLocator())
	ax.yaxis.set_major_locator(plt.NullLocator())
	ax.pcolormesh(X,Y,Z,norm=colors.LogNorm(vmin=0.1 , vmax=Z.max()+5) , cmap='gray')
	plt.tight_layout() ;

	if os.path.exists (pathname) :		# Remove the file if it already exists
		os.remove(pathname)

	plt.savefig(pathname,transparent = True, bbox_inches = 'tight', pad_inches = 0, dpi = DPI) #filename comes here
	plt.close (fig)

	############################################################ Reshaping

	rows = Z.shape[0] 
	columns = Z.shape[1]
	
	img = cv2.imread (pathname, cv2.IMREAD_UNCHANGED)
	os.remove (pathname)

	resized = cv2.resize (img, (columns, rows), interpolation = cv2.INTER_AREA)

	resized2 = np.ndarray (shape = (rows,columns))
	for i in range(0, rows) :
		resized2[rows-i-1] = resized[i,:,0]

	from PIL import Image
	im = Image.fromarray (resized2.astype(np.uint8))
	im.save(pathname)


def export_png_wrapper (filename , Z , csv_abs_path, param_test) :
	############################################################
	# Inputs - name of object
	#        - cutout data
	#        - dpi of screen
	#        - offset of screen
	#        - std value for any possible ML
	#        - parameter testing truth value. Overwrites existing
	#		   png file is true, else it doesn't.

	# Exports png files of original size of the cutout and of $std x $std
	############################################################

	rows,columns = Z.shape
	Y,X = np.mgrid[0:rows,0:columns]		# The X,Y grid axes will be passed to export_png ()

	for i in range (0,rows) :
		for j in range(0,columns) :
			if Z[i][j] < 0 :
				Z[i][j] = 0.1 * np.exp(-((Z[i][j]/0.1)**2))


	#########################################################################################
    #########################################################################################
    # Exporting original size png -

	png_og_path = csv_abs_path + "/Data/" + filename + "_og_size_cutout.png"
	
	if param_test :
		if os.path.exists(png_og_path) :
			os.remove(png_og_path)
		export_png (png_og_path , X , Y, Z , rows , columns, csv_abs_path)
	else :
		if not os.path.exists(png_og_path) : 
			export_png (png_og_path , X , Y, Z , rows , columns, csv_abs_path)
	


def smooth_png (pathname , save_pathname , sigma_x , sigma_y) :
	############################################################
	# Inputs - path of unsmoothed png file
	#        - path of smoothed png file
	#        - std deviation along x direction to smooth
	#        - std deviation along y direction to smooth
	#        - dpi of screen
	#        - offset of screen

	# Exports a smoothed png file
	############################################################


	img = cv2.imread(pathname)
	blur = cv2.GaussianBlur(img,(sigma_x,sigma_y),0)
	rows = img.shape[0]
	columns = img.shape[1]
	Y,X = np.mgrid[0:rows,0:columns]

	

	from PIL import Image
	im = Image.fromarray (blur.astype(np.uint8))
	im.save(save_pathname)


def smooth_png_wrapper (filename , sigma_x , sigma_y, csv_abs_path, param_test) :
	############################################################
	# Inputs - name of the object
	#        - std deviation along x direction to smooth
	#        - std deviation along y direction to smooth
	#        - dpi of screen
	#        - offset of screen
	#        - parameter testing truth value. Overwrites existing
	#		   png file is true, else it doesn't.
	############################################################

	# Exporting original size png -

	png_og_path = csv_abs_path + "/Data/" + filename + "_og_size_cutout.png"
	png_og_smooth_path = csv_abs_path + "/Data/" + filename + "_og_size_cutout_smooth.png"
	
	if param_test :
		if os.path.exists(png_og_smooth_path) :
			os.remove (png_og_smooth_path)
		smooth_png (png_og_path , png_og_smooth_path , sigma_x , sigma_y)
	else :
		if not os.path.exists(png_og_smooth_path) :
			smooth_png (png_og_path , png_og_smooth_path , sigma_x , sigma_y)
	
	

def smooth_contour (png_path , contour_path , r , c , levels) :
	############################################################
	# Inputs - path to smoothed png file
	#        - path to save contour
	#        - rows of contour file
	#        - columns of contour file
	#        - levels of contour
	#        - dpi of screen
	#        - offset of screen
	#        - xm parameter for image scaling control
	#        - xi parameter for image scaling control
	#        - ym parameter for image scaling control
	#        - yi parameter for image scaling control

	# Exports a contour image of the cutout
	############################################################


	img = cv2.imread (png_path)
	Z = img[:,:,0]
	rows = img.shape[0]
	columns = img.shape[1]
	Y,X = np.mgrid[0:rows,0:columns]

	# Flipping the png data read by imread ()
	Z_flip = np.ndarray (shape = (rows,columns))
	for i in range (0, rows) :
		Z_flip[i,:] = Z[rows-1-i]

	fig = plt.figure (figsize= (30,30)) 		# Setting figure size
	ax = fig.add_subplot (111)
	ax.axis('off')
	ax.margins(0,0)
	ax.xaxis.set_major_locator(plt.NullLocator())
	ax.yaxis.set_major_locator(plt.NullLocator())
	ax.contourf (X,Y,Z_flip,cmap='gray',levels=levels)
	
	plt.savefig(contour_path,transparent = True, bbox_inches = 'tight', pad_inches = 0)
	plt.close(fig)

	############################################################ Reshaping
	import peak_util as pku 
	pku = importlib.reload(pku)

	img = cv2.imread (contour_path, cv2.IMREAD_UNCHANGED)
	Z_ctr_uq = pku.contour_unique_val(img[:,:,0])

	if Z_ctr_uq[-1] != 255 :
		Z_ctr_uq.append(255)
	if Z_ctr_uq[0] != 0 :
		Z_ctr_uq.insert(0, 0)

	os.remove (contour_path)
	resized = cv2.resize (img, (columns, rows), interpolation = cv2.INTER_AREA)
	#print (resized.shape)
	

	for i in range (0, resized.shape[0]) :
		for j in range (0, resized.shape[1]) :
			k = 0 
			while k < len(Z_ctr_uq) :
				if Z_ctr_uq[k] > resized[i,j,0] :
					break 

				k = k + 1

			resized[i,j,0] = Z_ctr_uq[k-1]
			resized[i,j,1] = Z_ctr_uq[k-1]
			resized[i,j,2] = Z_ctr_uq[k-1]

	from PIL import Image
	im = Image.fromarray (resized.astype(np.uint8))
	im.save(contour_path)


def smooth_contour_wrapper (filename, Z , levels , csv_abs_path, param_test) :
	############################################################
	# Inputs - name of object
	#        - cutout data
	#        - levels of contour
	#        - dpi of screen
	#        - contour of screen
	#        - std size for any machine learning
	#        - xm parameter for image scaling control
	#        - xi parameter for image scaling control
	#        - ym parameter for image scaling control
	#        - yi parameter for image scaling control
	#        - parameter testing truth value. Overwrites existing
	#		   png file is true, else it doesn't.
	############################################################

	# Exporting original size png -

	png_og_smooth_path = csv_abs_path + "/Data/" + filename + "_og_size_cutout_smooth.png"
	contour_og_path = csv_abs_path + "/Data/" + filename + "_og_size_contour.png"


	if param_test :
		if os.path.exists (contour_og_path) :
			os.remove (contour_og_path)
		smooth_contour (png_og_smooth_path , contour_og_path , Z.shape[0] , Z.shape[1]  , 
						levels) 
	else :
		if not os.path.exists (contour_og_path) :
			smooth_contour (png_og_smooth_path , contour_og_path , Z.shape[0] , Z.shape[1] , 
							levels) 