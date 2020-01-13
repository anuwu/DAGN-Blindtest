import numpy as np
import matplotlib.pyplot as plt
import os 
import random

def get_neighbour_tuples (x,y) :
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
	neighbour_tuples = get_neighbour_tuples (x, y)

	val = 0 
	for (xn, yn) in neighbour_tuples :
		pix_val = cutout[xn][yn]
		if pix_val < 0 :
			pix_val = 0

		val = val + pix_val
	val = val/9 

	return val 

def thicc_mark (Z_flip, rows, x, y) :
	Z_flip[rows - x-1+1,y] = 0
	Z_flip[rows - x-1-1,y] = 0
	Z_flip[rows - x-1,y+1] = 0	
	Z_flip[rows - x-1,y-1] = 0
	Z_flip[rows - x-1+1,y+1] = 0
	Z_flip[rows - x-1-1,y+1] = 0
	Z_flip[rows - x-1-1,y-1] = 0
	Z_flip[rows - x-1+1,y-1] = 0

	return Z_flip

def peak_mark (Z_flip, rows, x, y, snr, cutout, done_file, thicc) :
	Z_flip[rows - x-1,y] = 0
	#print ("Peak 1 is at " + str((x, y)) + " with value = " + str(cutout[x][y]) + " and snr = " + str(snr))
	#done_file.write ("Peak 1 is at " + str((x, y)) + " with value = " + str(cutout[x][y]) + " and snr = " + str(snr))
	# Thick marking ...
	if thicc :
		Z_flip = thicc_mark (Z_flip, rows, x, y)

	return Z_flip

def double_snr_peak_mark (Z_flip, rows, px1, py1, snr1, px2, py2, snr2, cutout, done_file, thicc) :
	case_list = {1: 'Double', 2:'Single1', 3:'Single2', 4:'NoPeak'}
	if snr1 >= 3 and snr2 >= 3 :
		snr_case = 1
	elif snr1 >= 3 and not (snr2 >= 3) :
		snr_case = 2
	elif not (snr1 >= 3) and snr2 >= 3 :
		snr_case = 3
	else :
		snr_case = 4

	if case_list[snr_case] == 'Double' :
		Z_flip = peak_mark (Z_flip, rows, px1, py1, snr1, cutout, done_file, thicc)
		Z_flip = peak_mark (Z_flip, rows, px2, py2, snr2, cutout, done_file, thicc)
		peak_plot_ret = 'Double'  
	elif case_list[snr_case] == 'Single1' :
		Z_flip = peak_mark (Z_flip, rows, px1, py1, snr1, cutout, done_file, thicc)
		peak_plot_ret = 'Single' 
	elif case_list[snr_case] == 'Single2' :
		Z_flip = peak_mark (Z_flip, rows, px2, py2, snr2, cutout, done_file, thicc)
		peak_plot_ret = 'Single'
	elif case_list[snr_case] == 'NoPeak' :
		peak_plot_ret = 'NoPeak'

	return (Z_flip, peak_plot_ret)



def contour_unique_val (Z_ctr) :
	############################################################
	# Inputs - png data
	
	# Returns a list of the unique contour pixel values in 
	# ascending order.
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
	#		 - peak y
	#		 - list of connected regions
	
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

	border_problem = False 
	for tup in neighbor_tuples :
		if tup not in Z_regions :
			border_problem = True 
			break 

	return border_problem

def get_noise_cutout (peak_plot_name, levels, env_level, Z_ctr_uq, Z_ctr) :
	obj_path = peak_plot_name[:peak_plot_name.find('_')]
	fits_path = obj_path + "_cut.fits"

	from astropy.io import fits
	hdul = fits.open(fits_path , memmap = False)
	hdu = hdul[0]
	cutout = hdu.data

	actual_env_level = int(np.ceil(env_level * len(Z_ctr_uq)/levels))
	
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

	avg_noise = 0
	for i in range(0, 10) :
		centre_noise = random.choice (snr_noise_list)
		x, y = centre_noise

		rand_noise = get_avg_neighvals (x, y, cutout)
		avg_noise = avg_noise + rand_noise

	avg_noise = avg_noise/10

	return (avg_noise, cutout)

# Note - The algorithm can be improved by reworking the len(reg1) > len(reg2) condition. It's too strict
def peak_plot (peak_dist , Z_regions , Z_img , Z_ctr, Z_ctr_uq, levels, env_level, peak_plot_name , thicc, done_file) :
	############################################################
	# Inputs - peak histogram
	#		 - list of connected high regions
	#		 - png data
	#		 - name of the file to mark peaks
	#		 - thick marking or thin marking truth value

	# Exports one or two png files in which the peaks are marked
	# If there's only one peak, it exports a _only.png
	# If there is more than one peak, it exports an _all.png and
	# _top_pair.png file

	############################################################

	# Flipping png data read by imread ()
	Z_flip = np.ndarray (shape = Z_img.shape)
	rows = Z_img.shape[0]
	for i in range(0 , rows) :
		Z_flip[i] = Z_img[rows - i - 1] ;

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

	############################################################	

	avg_noise , cutout = get_noise_cutout (peak_plot_name, levels, env_level, Z_ctr_uq, Z_ctr)
	no_peak = len(peak_dist)

	if avg_noise == None :
		return "NoNoise"
	
	############################################################

	peak_plot_ret = 'Single'
	if no_peak >= 2 :
		# Selecting top two peaks with maximum intensity value as peak_dist is reverse sorted
		px1 = peak_dist[0][0]
		py1 = peak_dist[0][1]
		px2 = peak_dist[1][0]
		py2 = peak_dist[1][1]

		#Be careful about using this. Peak may exist at the boundary
		signal1 = get_avg_neighvals (px1,py1,cutout)
		signal2 = get_avg_neighvals (px2,py2,cutout)
		snr1 = signal1/avg_noise
		snr2 = signal2/avg_noise

		same_region = False
		peak_plot_ret = 'Double'
		reg1 = None
		reg2 = None

		# Finding the regions in which the two peaks belong ...
		for reg in Z_regions :
			if (px1 , py1) in reg :
				reg1 = reg 

			if (px2 , py2) in reg :
				#print ("Belongs 2nd peak")
				reg2 = reg 

			if (not reg1 == None) and (not reg2 == None) and reg1 == reg2 :
				same_region = True
				break 

		# If the two peaks are not in the same connected region, then they are less likely
		# to belong to the same envelope and hence the two peaks are reported as separate objects ...
		if not same_region :
			if (reg1 == None or reg2 == None) :
				#print ("Algorithm failed")
				return "Failed"

			peak_plot_ret = 'Single' 		# Truth value signifying that there is only one peak

			# Selecting region with larger number ofpoints ...
			if len(reg1) > len(reg2) :
				if snr1 >= 3 :
					Z_flip = peak_mark (Z_flip, rows, px1, py1, snr1, cutout, done_file, thicc)
				else :
					peak_plot_ret = 'NoPeak' 
					#return None 

			# Selecting region with larger number of points ...			
			elif len(reg2) > len(reg1) :
				if snr2 >= 3 :
					Z_flip = peak_mark (Z_flip, rows, px2, py2, snr2, cutout, done_file, thicc)
				else :
					peak_plot_ret = 'NoPeak'
					#return None

			# If regions have same length, then there are actually two peaks ...
			else :
				Z_flip, peak_plot_ret = double_snr_peak_mark (Z_flip, rows, px1, py1, snr1, px2, py2, snr2, cutout, done_file, thicc)
				#if peak_plot_ret == None :
					#return None 
		else :
			# This case is needed in case of crowding of peaks around a boundary

			border_problem_1 = border_problem_decide (px1 , py1 , Z_regions)		# If first peak is at boundary
			border_problem_2 = border_problem_decide (px2 , py2 , Z_regions)		# If second peak is at boundary

			if not border_problem_1 and border_problem_2 :
				if snr1 >= 3 :
					Z_flip = peak_mark (Z_flip, rows, px1, py1, snr1, cutout, done_file, thicc)
					peak_plot_ret = 'Single' 
				else :
					peak_plot_ret = 'NoPeak' 
					#return None
			elif border_problem_1 and not border_problem_2 :
				if snr2 >= 3 :
					Z_flip = peak_mark (Z_flip, rows, px2, py2, snr2, cutout, done_file, thicc)
					peak_plot_ret = 'Single' 
				else :
					peak_plot_ret = 'NoPeak'
					#return None
			else :	#No border problem. This is supposed to be the normal case
				Z_flip, peak_plot_ret = double_snr_peak_mark (Z_flip, rows, px1, py1, snr1, px2, py2, snr2, cutout, done_file, thicc)
				#if peak_plot_ret == None :
					#return None 

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
		px1 = peak_dist[0][0]
		py1 = peak_dist[0][1]
		signal1 = get_avg_neighvals (px1,py1,cutout)
		snr1 = signal1/avg_noise

		if snr1 >= 3 :
			Z_flip = peak_mark (Z_flip, rows, px1, py1, snr1, cutout, done_file, thicc)
		else :
			peak_plot_ret = 'NoPeak'

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
		peak_plot_ret = 'NoPeak'
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