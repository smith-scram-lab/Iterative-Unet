from migration_yz.migration import *
import numpy as np

filematrix = np.load('filematrix.npy')
m = migration(filematrix)
print(m.get_loc_history().shape)

