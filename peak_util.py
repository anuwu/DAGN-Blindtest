import numpy as np
import matplotlib.pyplot as plt
import os 
import random

def get_neighbour_tuples (x,y) :
	############################################################
	# Given a cartesian point, returns a list of its 8 neighbors
	############################################################
	neighbour_tuples = [ (x+1,y),
			     (x,y+1),
			     (x-1,y),
			     (x,y-1),
			     (x+1,y+1),
			     (x+1,y-1),
			     (x-1,y+1),
			     (x-1,y-1)
			   ]

	return neighbour_tuples

def get_avg_neighvals (x, y, cutout) :
	############################################################
	# Inputs - x coordinate
	#	 - y coorindate
	#	 - cutout data
	#
	# Output - Returns the average pixel values of the point and
	#	   its neighbors
	############################################################
	
	neighbour_tuples = get_neighbour_tuples (x, y)
	val = 0 
	for (xn, yn) in neighbour_tuples :
		pix_val = cutout[xn][yn]
		if pix_val < 0 :	#Negative pixel values replaced with 0
			pix_val = 0

		val = val + pix_val
	val = val/9 

	return val 

def thicc_mark (Z_flip, rows, x, y) :
	############################################################
	# Inputs - Flipped 2d image data to be saved.
	#	 - number of rows of image
	#	 - x coordinate at which to mark peak
	#	 - y coordinate arounch which to mark peak
	#
	# Output - Returns the flipped matrix after thick marking
	############################################################
	Z_flip[rows - x-1+1,y] = 0
	Z_flip[rows - x-1-1,y] = 0
	Z_flip[rows - x-1,y+1] = 0	
	Z_flip[rows - x-1,y-1] = 0
	Z_flip[rows - x-1+1,y+1] = 0
	Z_flip[rows - x-1-1,y+1] = 0
	Z_flip[rows - x-1-1,y-1] = 0
	Z_flip[rows - x-1+1,y-1] = 0

	return Z_flip

def peak_mark (Z_flip, rows, x, y, cutout, done_file, thicc) :
	############################################################
	# Inputs - Flipped image data to be saved
	#	 - Number of rows of image
	#	 - x coordinate of peak
	#	 - y coordinate of peak
	#	 - cutout data
	#	 - log file handler for any future purposes
	#	 - thicc marking boolean value
	#
	# Output - Outputs the flipped image data after marking
	############################################################
	Z_flip[rows - x-1,y] = 0
	if thicc :
		Z_flip = thicc_mark (Z_flip, rows, x, y)

	return Z_flip

def double_snr_peak_mark (Z_flip, rows, px1, py1, snr1, px2, py2, snr2, cutout, done_file, thicc) :
	############################################################
	# Inputs - Flipped image data
	#	 - Number of rows of flipped image data
	#	 - x-coordinate of peak 1
	#	 - y-coordinate of peak 1
	#	 - SNR of peak 1
	#	 - x-coordinate of peak 2
	#	 - y-coordinate of peak 2
	#	 - SNR of peak 2
	#	 - cutout data
	#	 - log file handler for future purposes
	# 	 - thicc marking boolean value
	#
	# Output - Returns the flipped image data post peak marking
	#	   and undergoing SNR checks.
	############################################################
	
	
	
	case_list = {1: 'Double', 2:'Single1', 3:'Single2', 4:'NoPeak'}
	if snr1 >= 3 and snr2 >= 3 :
		snr_case = 1
	elif snr1 >= 3 and not (snr2 >= 3) :
		snr_case = 2
	elif not (snr1 >= 3) and snr2 >= 3 :
		snr_case = 3
	else :
		snr_case = 4
		
	############################################################
	# The reasoning behind the conditions in this function is that 
	# a detected peak is not a legitimate peak if it has an SNR 
	# lesser than 3.
	############################################################

	if case_list[snr_case] == 'Double' :
		Z_flip = peak_mark (Z_flip, rows, px1, py1, cutout, done_file, thicc)
		Z_flip = peak_mark (Z_flip, rows, px2, py2, cutout, done_file, thicc)
		peak_plot_ret = 'Double'  
	elif case_list[snr_case] == 'Single1' :
		Z_flip = peak_mark (Z_flip, rows, px1, py1, cutout, done_file, thicc)
		peak_plot_ret = 'Single' 
	elif case_list[snr_case] == 'Single2' :
		Z_flip = peak_mark (Z_flip, rows, px2, py2, cutout, done_file, thicc)
		peak_plot_ret = 'Single'
	elif case_list[snr_case] == 'NoPeak' :
		peak_plot_ret = 'NoPeak'

	return (Z_flip, peak_plot_ret)



def contour_unique_val (Z_ctr) :
	############################################################
	# Inputs - png data
	
	# Output - Returns a list of the unique contour pixel values in 
	# 	   ascending order.
	############################################################
	Z_ctr_uq = []

	for i in range(0,Z_ctr.shape[0]) :
		for j in range(0,Z_ctr.shape[1]) :
			if not Z_ctr[i,j] in Z_ctr_uq :
				Z_ctr_uq.append(Z_ctr[i,j])

	Z_ctr_uq.sort ()

	return Z_ctr_uq 

def border_problem_decide (x , y , Z_regions) :
	############################################################
	# Inputs - peak x
	#	 - peak y
	#	 - list of connected regions
	
	# Returns True if reported peak is at the edge of a region 
	# Else it returns False.
	############################################################
	neighbor_tuples = [ (x+1,y),
			    (x,y+1),
			    (x-1,y),
			    (x,y-1),
			    (x+1,y+1),
			    (x+1,y-1),
		            (x-1,y+1),
			    (x-1,y-1)
			  ]
	
	############################################################
	# The reasoning is self-explanatory. If any coordinate point 
	# neighbouring a detected peak does not lie in the high-contour
	# region, it would have arisen out of edge-crowding.
	############################################################

	border_problem = False 
	for tup in neighbor_tuples :
		if tup not in Z_regions :
			border_problem = True 
			break 

	return border_problem


def get_noise_cutout (peak_plot_name, levels, env_level, Z_ctr_uq, Z_ctr) :
	############################################################
	# Inputs - To-be path of peak plot image (except appended flags)
	#	 - levels of contour
	#	 - environment level
	#	 - actual pixel value of contour levels
	#	 - contour image data
	#
	# Output - Returns the average noise level and cutout data
	############################################################
	
	# Extracting path of FITS file
	obj_path = peak_plot_name[:peak_plot_name.find('_')]
	fits_path = obj_path + "_cut.fits"

	from astropy.io import fits
	hdul = fits.open(fits_path , memmap = False)
	hdu = hdul[0]
	cutout = hdu.data

	# Rescaling actual environment level
	actual_env_level = int(np.ceil(env_level * len(Z_ctr_uq)/levels))
	
	############################################################
	# The logic to the part below is the following. It aims to find
	# a list of points which can be considered as noise and do not
	# lie at the boundary of the FITS cutout.
	#
	# Concentrate on the code inside the while loop -
	# For a certain value of i, noise_ctr_val is set to the pixel 
	# value of the ith contour level. noise_list is a list that 
	# picks out those points that have a pixel value equal to 
	# noise_ctr_val
	#
	# Among the points in noise_list, those that are not on the
	# boundary of the FITS cutout are appended into the list
	# noise_list_noboundary
	#
	# Further, among this list, those points whose one of 8 neighbors
	# belongs to a higher contour level than the ith level are discarded.
	# The points that remain are put into lowest_noise_list. This is
	# conceptually the noise list we are seeking.
	#
	# It could be possible that after this heavily filtration process,
	# lowest_noise_list ends up being an enpty list. In such a case
	# the final noise list is set to noise_list_noboundary.
	#
	# If this too is empty, then i is incremented by 1 and the loop
	# executes again. This repeats until i reaches that pixel value
	# which defines the environment level after proper rescaling
	# (held in variable actual_env_level). At this point, we conclude
	# that no reliable noise data can be extracted from the image
	# and SNR checks have failed.
	#
	# This is what results in the 'NoNoise' label in the output csv
	# file of the pipeline. However, this is rarely expected to occur
	# Even if it does, the object most probably has no peaks in actuality.
	############################################################
	
	i = 0
	while True :
		if i == actual_env_level :
			return (None, cutout)

		noise_ctr_val = Z_ctr_uq[i]
		noise_list = []
		noise_list_noboundary = []
		lowest_noise_list = []
		for (x, y), element in np.ndenumerate (Z_ctr) :
			if Z_ctr[x][y] == noise_ctr_val :
				noise_list.append((x,y))

				if not (x < 1 or x > (cutout.shape[0]-2) or y < 1 or y > (cutout.shape[1]-2)) :
					noise_list_noboundary.append((x,y))

					neighbour_tuples = get_neighbour_tuples(x,y)
					lowest_cond = True 				
					for (x, y) in neighbour_tuples :
						if Z_ctr[x][y] in Z_ctr_uq[i+1:] :
							lowest_cond = False

					if lowest_cond :
						lowest_noise_list.append((x,y))

		if len(lowest_noise_list) == 0 :
			snr_noise_list = noise_list_noboundary
		else :
			snr_noise_list = lowest_noise_list

		if len(snr_noise_list) == 0 :
			i = i + 1 
		else :
			break

	############################################################
	# The noise list is randomly sampled 10 times and average noise
	# value is calculated. Then an ensemble average is returned.
	############################################################

	avg_noise = 0
	for i in range(0, 10) :
		centre_noise = random.choice (snr_noise_list)
		x, y = centre_noise

		rand_noise = get_avg_neighvals (x, y, cutout)
		avg_noise = avg_noise + rand_noise

	avg_noise = avg_noise/10

	return (avg_noise, cutout)



# Note - The algorithm can be improved by reworking the len(reg1) > len(reg2) condition. It's too strict, but lite?
def peak_plot (peak_dist , Z_regions , Z_img , Z_ctr, Z_ctr_uq, levels, env_level, peak_plot_name , thicc, done_file) :
	############################################################
	# Inputs - peak histogram
	#	 - list of connected high regions
	#	 - smooth image data
	#	 - contour image data
	#	 - contour level pixel values
	#	 - levels of contour
	#	 - environment level
	#	 - name of file that saves that peak plot
	#	 -  thick marking or thin marking truth value

	# Output - Exports one or two png files in which the peaks are marked
	# 	   If there's only one peak, it exports a file whose name
	#	   is the objID and is appended by '_only.png'
	#	   For a double peak, it is '_top_pair.png'
	#	   For no deteced peak, it is '_none.png'
	#	 - Returns a string that contains the nature of the peak
	############################################################

	# Flipping png data read by imread ()
	Z_flip = np.ndarray (shape = Z_img.shape)
	rows = Z_img.shape[0]
	for i in range(0 , rows) :
		Z_flip[i] = Z_img[rows - i - 1]
		
	############################################################
	# The following condition traces back to env_level_contour_search
	# in the env_level_peak_search.py
	# In that module, if the list of high contour regions is empty,
	# then gradient ascent is skipped altogether.
	#
	# Thus there is no distinct signal to begin with and no peaks.
	############################################################
	if len(Z_regions) == 0 :
		fig = plt.figure ()
		ax = fig.add_subplot (111)

		ax.axis('off')
		ax.margins(0,0)
		ax.xaxis.set_major_locator(plt.NullLocator())
		ax.yaxis.set_major_locator(plt.NullLocator())
		plt.imshow (Z_flip , origin = 'lower', cmap='PuBu_r')

		plt.savefig (peak_plot_name + "_none.png" ,transparent = True, bbox_inches = 'tight', pad_inches = 0)
		plt.close (fig)

		return "NoPeak"

	# Obtain the average noise and cutout
	avg_noise , cutout = get_noise_cutout (peak_plot_name, levels, env_level, Z_ctr_uq, Z_ctr)
	no_peak = len(peak_dist)

	# Check the definition of get_noise_cutout for the 'NoNoise' condition.
	if avg_noise == None :
		return "NoNoise"
	
	############################################################
	# The large section of code-below is the heart of the decision
	# making process to determine the number of peaks in the object
	# image.
	############################################################

	peak_plot_ret = 'Single'
	if no_peak >= 2 :
		# Selecting top two peaks with maximum intensity value as peak_dist is reverse sorted
		px1 = peak_dist[0][0]
		py1 = peak_dist[0][1]
		px2 = peak_dist[1][0]
		py2 = peak_dist[1][1]

		#Be careful about using this. Peak may exist at the boundary
		# Calculating peak average signal values and SNR.
		signal1 = get_avg_neighvals (px1,py1,cutout)
		signal2 = get_avg_neighvals (px2,py2,cutout)
		snr1 = signal1/avg_noise
		snr2 = signal2/avg_noise

		same_region = False
		peak_plot_ret = 'Double'
		reg1 = None
		reg2 = None

		# Finding the regions in which the two peaks belong
		for reg in Z_regions :
			if (px1 , py1) in reg :
				reg1 = reg 

			if (px2 , py2) in reg :
				#print ("Belongs 2nd peak")
				reg2 = reg 

			if (not reg1 == None) and (not reg2 == None) and reg1 == reg2 :
				same_region = True
				break 
		
		############################################################
		# If the two peaks are not in the same connected region, then 
		# they are less likely to belong to the same envelope and only
		# one of them is the true single peak depending on which one of them
		# belongs to a larger connected region.
		############################################################
		if not same_region :
			
			# This is no expected to occur, but it happened once. Should be very rare.
			if (reg1 == None or reg2 == None) :
				return "Failed"

			peak_plot_ret = 'Single' 		# Truth value signifying that there is only one peak

			# Selecting region with larger number of points
			if len(reg1) > len(reg2) :
				if snr1 >= 3 : # Mark peak only if SNR is greater than 3.
					Z_flip = peak_mark (Z_flip, rows, px1, py1, cutout, done_file, thicc)
				else :	# Otherwise there is no peak
					peak_plot_ret = 'NoPeak' 

			# Selecting region with larger number of points			
			elif len(reg2) > len(reg1) :
				if snr2 >= 3 : # Mark peak only if SNR is greater than 3.
					Z_flip = peak_mark (Z_flip, rows, px2, py2, cutout, done_file, thicc)
				else : # Otherwise there is no peak
					peak_plot_ret = 'NoPeak'
					#return None

			# If regions have same length, then there are actually two peaks.
			else :
				Z_flip, peak_plot_ret = double_snr_peak_mark (Z_flip, rows, px1, py1, snr1, px2, py2, snr2, 
									      cutout, done_file, thicc)
		else :
			# This case is needed in case of crowding of peaks around a boundary

			border_problem_1 = border_problem_decide (px1 , py1 , Z_regions)		
			border_problem_2 = border_problem_decide (px2 , py2 , Z_regions)		

			if not border_problem_1 and border_problem_2 :
				if snr1 >= 3 : # Mark peak only if SNR is greater than 3.
					Z_flip = peak_mark (Z_flip, rows, px1, py1, cutout, done_file, thicc)
					peak_plot_ret = 'Single' 
				else : # Otherwise there is no peak
					peak_plot_ret = 'NoPeak' 
					#return None
			elif border_problem_1 and not border_problem_2 :
				if snr2 >= 3 : # Mark peak only if SNR is greater than 3.
					Z_flip = peak_mark (Z_flip, rows, px2, py2, cutout, done_file, thicc)
					peak_plot_ret = 'Single' 
				else : # Otherwise there is no peak
					peak_plot_ret = 'NoPeak'
					#return None
			else :	#No border problem. This is supposed to be the normal case
				Z_flip, peak_plot_ret = double_snr_peak_mark (Z_flip, rows, px1, py1, snr1, px2, py2, snr2, 
									      cutout, done_file, thicc)

		# Creating plots after deciding on nature of peaks
		fig = plt.figure ()
		ax = fig.add_subplot (111)
		ax.axis('off')
		ax.margins(0,0)
		ax.xaxis.set_major_locator(plt.NullLocator())
		ax.yaxis.set_major_locator(plt.NullLocator())
		plt.imshow (Z_flip , origin = 'lower', cmap='PuBu_r')

		if peak_plot_ret == 'Single' :
			plt.savefig (peak_plot_name + "_only.png" ,transparent = True, bbox_inches = 'tight', pad_inches = 0)
		elif peak_plot_ret  == 'Double' :
			plt.savefig (peak_plot_name + "_top_pair.png" ,transparent = True, bbox_inches = 'tight', pad_inches = 0)
		else :
			plt.savefig (peak_plot_name + "_none.png" ,transparent = True, bbox_inches = 'tight', pad_inches = 0)

		plt.close (fig)
	elif no_peak == 1 :	
		# Finding peak points
		px1 = peak_dist[0][0]
		py1 = peak_dist[0][1]
		
		# Finding peak signal value and SNR
		signal1 = get_avg_neighvals (px1,py1,cutout)
		snr1 = signal1/avg_noise

		if snr1 >= 3 : # Mark peak only if SNR is greater than 3.
			Z_flip = peak_mark (Z_flip, rows, px1, py1, cutout, done_file, thicc)
		else : #  Otherwise there is no peak
			peak_plot_ret = 'NoPeak'

		# Creating plots
		fig = plt.figure ()
		ax = fig.add_subplot (111)
		ax.axis('off')
		ax.margins(0,0)
		ax.xaxis.set_major_locator(plt.NullLocator())
		ax.yaxis.set_major_locator(plt.NullLocator())
		plt.imshow (Z_flip , origin = 'lower', cmap='PuBu_r')

		if peak_plot_ret == 'Single' :
			plt.savefig (peak_plot_name + "_only.png" ,transparent = True, bbox_inches = 'tight', pad_inches = 0)
		else :
			plt.savefig (peak_plot_name + "_none.png" ,transparent = True, bbox_inches = 'tight', pad_inches = 0)
		plt.close (fig)
	else :
		# There is no peak
		peak_plot_ret = 'NoPeak'
		
		# Creating plots
		fig = plt.figure ()
		ax = fig.add_subplot (111)
		ax.axis('off')
		ax.margins(0,0)
		ax.xaxis.set_major_locator(plt.NullLocator())
		ax.yaxis.set_major_locator(plt.NullLocator())
		plt.imshow (Z_flip , origin = 'lower', cmap='PuBu_r')

		plt.savefig (peak_plot_name + "_none.png" ,transparent = True, bbox_inches = 'tight', pad_inches = 0)
		plt.close (fig)


	return peak_plot_ret
