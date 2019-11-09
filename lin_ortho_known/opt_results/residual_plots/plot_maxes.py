import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
homeuser = os.path.expanduser('~')

filenames = ['residuals_blue00.npy',
             'residuals_blue01.npy',
             'residuals_blue02.npy',
             'residuals_blue03.npy']
datanames = ['blue00.npy',
             'blue01_rotated_90.npy',
             'blue02_rotated_90.npy',
             'blue03.npy']

perfx = np.linspace(-100, 100, 100)
perfy = np.zeros(100)

for i in range(0, 4):
# for residual_file in filenames:
    resids = np.load(filenames[i], allow_pickle=True)
    print(resids.shape)
    data = np.load(os.path.join(homeuser, datanames[i]), allow_pickle=True)

    my_shape = resids.shape
    # compute dx max, compute dy max, compute dz max
    dx_max = []
    dy_max = []
    dz_max = []
    for j in range(my_shape[1]):
            dx_max.append(resids[0][j].max())
            dy_max.append(resids[1][j].max())
            dz_max.append(resids[2][j].max())
    plt.figure()
    plt.plot(data[:, 1], dz_max, '-.')
    plt.xlabel('Pressure')
    plt.ylabel(r'$\Delta x max$')
    plt.show()
            # break
        # break
    # break