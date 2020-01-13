# The graph nodes (grid points) have already been appended with color.

def dfs_visit (Z , point) :
	############################################################
	# Inputs - png data
	#        - point from which to continue Depth First Search
	############################################################

	ind = Z.index(point)
	Z[ind] = (point[0] , point[1] , 'g')

	x = point[0] 
	y = point[1]

	# Adjacency list
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

		# Need this loop to check if adjacent points exist within the search region ...
		for z_tup in Z :
			if tup[0] == z_tup[0] and tup[1] == z_tup[1] and z_tup[2] == 'w' :
				dfs_visit (Z , z_tup) 

	Z[ind] = (point[0] , point[1] , 'b')		# blacken the point after adjacency list has been exhausted


def get_regions (Z_search) :
	############################################################
	# Inputs - search region

	# Returns a lit of connected regions
	############################################################

	regions = []	# Initial list of connected regions is empty
	Z = []

	# Append each point with color.
	for point in Z_search :
		Z.append(point + ('w',))

	for point in Z :

		if point[2] == 'w' :
			dfs_visit (Z , point)

			# Each connected component starts as an empty list
			reg = []

			i = 0 
			while (i < len(Z)) :
				if Z[i][2] == 'b' :
					reg.append ((Z[i][0] , Z[i][1]))
					Z.remove(Z[i])
					i = i - 1		# Removing element from list pushes the list backward. Resetting i value to correct position.

				i = i + 1 

			# Append current region to list of regions
			regions.append(reg)

	return regions