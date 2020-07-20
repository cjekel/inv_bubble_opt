import os
import numpy as np
from scipy.spatial.distance import cdist

homeuser = os.path.expanduser('~')

datanames = ['blue00.npy',
             'blue01_rotated_90.npy',
             'blue02_rotated_90.npy',
             'blue03.npy']
# In [22]: xy[0, ind_max_disp_x, :]
# Out[22]: array([56.25,  0.  ,  0.  ])

# In [23]: xy[0, ind_max_disp_y, :]
# Out[23]: array([ 0. , 62.5,  0. ])

# In [24]: xy[0, ind_max_disp_z, :]
# Out[24]: array([0., 0., 0.])
max_dx = np.array([[56.25, 0.0]])
max_dy = np.array([[0.0, 62.5]])
max_dz = np.array([[0.0, 0.0]])
for i in range(4):
    dataset = np.load(os.path.join(homeuser, datanames[i]), allow_pickle=True)
    new_dataset = []
    for j in range(len(dataset)):
        xyz_dataset = np.empty((3, 6))
        xyz_dataset.fill(np.nan)
        p = dataset[j, 1]
        # find the data points closes to the max_x, max_y, and max_z
        # deflection locations
        xy = dataset[j, 0][:, :2]
        d = cdist(xy, max_dx)
        # set the xy location and grab the displacement value
        # dx
        d_ind = np.argmin(d)
        xyz_dataset[0, :4] = dataset[j, 0][d_ind, :4]

        # set the xy location and grab the displacement value
        # dy
        d = cdist(xy, max_dy)
        d_ind = np.argmin(d)
        xyz_dataset[1, :3] = dataset[j, 0][d_ind, :3]
        xyz_dataset[1, 4] = dataset[j, 0][d_ind, 4]

        # set the xy location and grab the displacement value
        # dz
        d = cdist(xy, max_dz)
        d_ind = np.argmin(d)
        xyz_dataset[2, :3] = dataset[j, 0][d_ind, :3]
        xyz_dataset[2, 5] = dataset[j, 0][d_ind, 5]
        new_dataset.append([xyz_dataset, p])

    np.save('new_max_only_' + datanames[i], new_dataset)
