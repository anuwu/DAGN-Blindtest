import cv2
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

import dfs
import grad_asc as grad
import peak_util as pku

import importlib


def env_level_contour_search (png_path , contour_path , peak_plot , background_ratio , 
			levels, env_level ,iters , neigh_tol) :
	############################################################
	# Inputs - path of smoothed image
	#        - path of contour image
	#        - name of the file where peaks will be indicated
	#	 - background_ratio value. If 0, use env_level, if not, 
	#          use background_ratio * levels for envelope detection
	# 	 - total contour levels
	#	 - environment level
	#	 - iterations to run gradient ascent for
	#	 - neighborhood tolerance below which peaks are identical
	#
	# Output - Returns - peak distribution
	#		   - smoothed cutout image data
	#		   - region of high contour
	#		   - contour image data
	#		   - list of pixel values corresponding to each
	#		     contour level
	############################################################


	import dfs
	import grad_asc as grad
	import peak_util as pku
	pku = importlib.reload (pku)
	grad = importlib.reload(grad)
	dfs = importlib.reload(dfs)

	############################################################
	# (i,j) represents the png coordinate
	# To shift to cutout coordinates for visualisation
	# (i , j) ---> (rows - i , j)
	############################################################

	# use background_ratio value only if it isn't zero ...
	if not background_ratio == 0 :
		env_level = int(background_ratio*levels)

	# Reading smoothed image ...
	smooth_img = cv2.imread(png_path)
	rows = smooth_img.shape[0]
	columns = smooth_img.shape[1]
	Z_img = smooth_img[:,:,0]

	# Reading contour ...
	contour_img = cv2.imread(contour_path)
	Z_ctr = contour_img[:,:,0]
	
	############################################################
	# matplotlib's contourf() function in smooth_contour() in
	# cutout.py does not always generate the number of levels as 
	# passed to it. It depends on the nature of the input image fed 
	# to it. Thus, we need to deterine the actual number of contour
	# levels using pku.contour_unique_val and then scale the 
	# environment level accordingly.
	############################################################

	# Contains $level number of pixel values corresponding to each level
	Z_ctr_uq = pku.contour_unique_val(Z_ctr)	

	# Rescaling actual environment level
	actual_env_level = int(np.ceil(env_level * len(Z_ctr_uq)/levels))

	# Constructing a list of tuples that lie in high contour region 
	Z_high = []			
	for i in range (0 , rows) :
		for j in range (0 , columns) :
			if Z_ctr[i,j] > Z_ctr_uq[actual_env_level - 1] :
				#print (i, j, Z_ctr[i,j])
				tup = (i , j)
				Z_high.append(tup)

	if len(Z_high) == 0 :
		return (None, Z_img, [], Z_ctr, Z_ctr_uq)
	
	# run gradient ascent $iters times and obtain histogram of peaks
	peak_dist = grad.grad_asc (Z_img , Z_high , iters , neigh_tol)	
	
	# Find connected components of Z_high to classify peaks later
	Z_high_regions = dfs.get_regions (Z_high)		

	return (peak_dist , Z_img , Z_high_regions, Z_ctr, Z_ctr_uq)	


def env_level_peak_plot (filename, background_ratio , 
			levels, env_level , iters , neigh_tol , thicc , 
			 done_file, csv_abs_path, param_test) :
	############################################################
	# Inputs - object name
	#	 - background_ratio value. If 0, use env_level, if not, 
	#	   use background_ratio * levels for envelope detection
	# 	 - total contour levels
	#	 - environment level
	#	 - iterations to run gradient ascent for
	#	 - neighborhood tolerance below which peaks are identical
	#	 - Truth value to thinly mark peak, or thickly mark it
	#	 - file handler of logfile for any future development
	#	 - path of saving the peak plots
	#        - parameter testing truth value. Overwrites existing
	#	   png file is true, else it doesn't.
	#
	# Output - Returns a string that denotes whether the object had
	#	   1. Single Peak
	#	   2. Double Peak
	#	   3. NoPeak
	#	   4. NoNoise (Hence don't implement SNR. Mostly no peak)
	#	   5. Failure (Can occur in the RAREST cases)
	############################################################


	import peak_util as pku
	pku = importlib.reload (pku)

	is_only_og = None 

	png_og_smooth_path = csv_abs_path + "/Data/" + filename + "_og_size_cutout_smooth.png"
	contour_og_path = csv_abs_path + "/Data/" + filename + "_og_size_contour.png"
	peak_plot_og = csv_abs_path + "/Data/" + filename + "_og_size_env_peaks"

	############################################################
	# Checking whether files already exists or not, and taking 
	# appropriate decision with respect to param_test
	############################################################

	if param_test :
		# Remove file if any of them are present ...
		if os.path.exists (peak_plot_og + "_top_pair.png") :
			os.remove(peak_plot_og + "_top_pair.png")
		if os.path.exists (peak_plot_og + "_only.png") :
			os.remove(peak_plot_og + "_only.png")
		if os.path.exists (peak_plot_og + "_none.png") :
			os.remove (peak_plot_og + "_none.png")

		# Obtain the peak distribution, smoothed data, and list of connected components ...
		peak_dist , Z_img , Z_regions, Z_ctr, Z_ctr_uq = env_level_contour_search (png_og_smooth_path , 
											   contour_og_path , peak_plot_og , 
											   background_ratio , levels, 
											   env_level ,iters , neigh_tol)

		# Plot the peaks and obtain the truth value for single peak ...
		is_only_og = pku.peak_plot (peak_dist , Z_regions , Z_img , Z_ctr, Z_ctr_uq, 
					    levels, env_level, peak_plot_og , True, done_file)
	else :
		# If any of the peak plot types do not exist -
		if not (os.path.exists (peak_plot_og + "_top_pair.png") or os.path.exists (peak_plot_og + "_only.png") or os.path.exists (peak_plot_og + "_none.png")) :
			
			# Obtain the peak distribution, smoothed data, and list of connected components ...
			peak_dist , Z_img , Z_regions, Z_ctr, Z_ctr_uq = env_level_contour_search (png_og_smooth_path , 
												   contour_og_path , peak_plot_og ,
												   background_ratio , levels, 
												   env_level ,iters , neigh_tol)


			# Plot the peaks and obtain the truth value for single peak ...
			is_only_og = pku.peak_plot(peak_dist , Z_regions , Z_img , Z_ctr, Z_ctr_uq, 
						   levels, env_level, peak_plot_og , False, done_file)
		

	return is_only_og
