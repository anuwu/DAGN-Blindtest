import numpy as np

# Two dimensional multi-indexer
coodIndexer = lambda x : (lambda f : (f(x[:,0]), f(x[:,1])))\
                        (lambda y : [] if not y.size else y)
# Boolean array for equality within region
ptRegCheck = lambda pt, reg : (pt == reg).all(axis=1)
# Check if point is in a region
isPointIn = lambda pt, reg : ptRegCheck(pt, reg).any()
# Return index of a point in a region
ptIndex = lambda pt, reg : np.argwhere(ptRegCheck(pt, reg))

# Returns a (2t x 2t) neighborhood centred around a pixel
tolNeighs = lambda pt, t : [(pt[0]+dx, pt[1]+dy)
                        for dx in range(-t, t+1) for dy in range(-t, t+1)
                        if dx or dy]
# Filters the output of the above within a region 'reg'
neighsInReg = lambda pt, reg, t : [p for p in tolNeighs(pt, t)
                                if isPointIn(p, reg)]
# Returns immediate neighbors needed by DFS within a region 'reg'
dfsNeighs = lambda ind, reg : [ptIndex(pt, reg)[0][0] for pt in
                            neighsInReg(reg[ind], reg, 1)
                            if ptIndex(pt, reg).size > 0]
