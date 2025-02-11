import rpy2.robjects as ro
import numpy as np
from scipy.sparse import csc_matrix

r_obj = ro.r['readRDS']('/users/hjiang/GenoDistance/Data/raw_count.rds')

# Extract matrix dimensions
dims = list(r_obj.slots['Dim'])  # [nrows, ncols]

# Extract the slot data needed to rebuild the matrix
i = np.array(r_obj.slots['i'], dtype=np.int32)
p = np.array(r_obj.slots['p'], dtype=np.int32)
x = np.array(r_obj.slots['x'], dtype=np.float64)

# Create a SciPy sparse csc_matrix
sparse_mat = csc_matrix((x, i, p), shape=(dims[0], dims[1]))

# Now you can use sparse_mat in Python
print(sparse_mat.shape)
print(sparse_mat)
