import numpy as np

def max_neighbor (Z_img , Z_search_reg , x , y) :
	############################################################
	# Inputs - png data of the smoothed image
	#        - List of tuples 
	#        - search region for the peaks
	#		 - x point whose gradient is to be calculated
	# 		 - y point whose gradient is to be calculated

	# Returns the neighbouring tuple with maximum value.
	############################################################

	centre_val = int(Z_img[x,y])		# Pixel value of chosen point (x , y)

	# A list of the neighbouring points ...
	neighbor_tuples = [ (x+1,y),
						(x,y+1),
						(x-1,y),
						(x,y-1),
						(x+1,y+1),
						(x+1,y-1),
						(x-1,y+1),
						(x-1,y-1)
						]


	neighbor_vals = []		# A list that contains the pixel values of the neighbouring tuples

	# Appends pixel value if the neighbour lies within the search region, else None ...
	for tup in neighbor_tuples :
		if (tup[0] >= Z_img.shape[0] or tup[1] >= Z_img.shape[1]) :
			neighbor_vals.append(None)
		elif tup in Z_search_reg :
			neighbor_vals.append(int(Z_img[tup[0] , tup[1]]))
		else :
			neighbor_vals.append(None)

	
	# Calculates the neighbouring pixel whose value maximally differs from the chosen point ...
	i = 0 
	max_neigh_diff = np.NINF
	max_tup = (None , None)
	for tup in neighbor_tuples :
		val = neighbor_vals[i]
		if (not val == None) and val -  centre_val > max_neigh_diff :
			max_neigh_diff =  val -  centre_val
			max_tup = tup
		i = i + 1

	# Note that this difference in pixel value can be negative, in which case, the chosen point is already the maxima ...
	if max_neigh_diff <= 0 :
		max_tup = (x , y)

	return max_tup

def is_neighbour (x1 , y1 , x2 , y2 , radius) :
	############################################################
	# Inputs - x1
	#        - y1
	#        - x2
	#		 - y2 
	# 		 - Neighbourhood tolerance value below which the points
	#		 - (x1 , y1) and (x2 , y2) are neighbours

	# Returns True if they are neighbors, else False.
	############################################################
	x = abs(x2 - x1)
	y = abs(y2 - y1)

	dist = np.sqrt(x**2 + y**2)

	if dist <= radius :
		return True 
	else :
		return False 


def grad_asc (Z_img , Z_search_reg , iters , neigh_tol) :
	############################################################
	# Inputs - png data of smoothed file
	#        - regions where gradient ascent has to be conducted
	#        - number of iterations for which gradient ascent has 
	# 		   to be run
	#        - minimum pixel value separation below which two peaks 
	# 		   cannot be resolved

	# Returns a histogram of peak distribution after gradient ascent
	# is run $iters times.
	############################################################

	# Will contain all valid peaks and their corresponding pixel 
	# values after gradient ascent is run for $iters times ... 
	peaks = []		

	for i in range(0 , iters) :
		# Randomly initialising start point within search region ...
		xold , yold = Z_search_reg[np.random.randint(len(Z_search_reg) , size = 1)[0]]		

		while True :	
			xnew , ynew = max_neighbor (Z_img , Z_search_reg , xold , yold)		# Finding the neighbour with maximum value a.k.a gradient

			if xnew == xold and ynew == yold :		# Current point is maxima
				break 
			else :
				xold = xnew 
				yold = ynew

		#Maxima detected can be at border, in which case it is not a true maxima ...	
		if not is_boundary_peak (xnew , ynew , Z_search_reg) :		
			peaks.append((xnew,ynew,Z_img[xnew,ynew]))	

	############################################################################################################
	############################################################################################################

	peak_uq = []		# Will contain only the unique peaks
	peak_freq = []		# Histogram of peaks

	# Loop to find unique peaks ...
	for p in peaks :
		uq_exist = False 
		for i in range(0 , len(peak_uq)) :

			#If two peaks are at neighborhood tolerance distance, they are essentially the same peak ...
			if is_neighbour (peak_uq[i][0] , peak_uq[i][1] , p[0] , p[1] , neigh_tol) :		
				peak_freq[i] = peak_freq[i] + 1
				uq_exist = True 
				break 

		if not uq_exist :
			peak_uq.append(p) 
			peak_freq.append(1)			# Start off histogram with frequency value of one 

	# Appending the frequency value to the peak_uq list ...
	for i in range(0 , len(peak_uq)) :
		peak_uq[i] = peak_uq[i] + (peak_freq[i],) 		

	# Sorting in descending order of pixel value ...
	peak_dist = sorted(peak_uq , key=lambda peak_uq:peak_uq[2] , reverse = True)
	return peak_dist


def is_boundary_peak (x , y , Z_region) :
	############################################################
	# Inputs - x point
	#        - y point
	#        - search region for the peaks

	# Returns True if detected maxima is at the boundary of the
	# search region, eles False. 
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

	for tup in neighbor_tuples :
		if tup not in Z_region :
			return True

	return False
