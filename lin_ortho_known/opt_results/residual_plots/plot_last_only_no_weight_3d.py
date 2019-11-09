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
    for j in range(my_shape[1]):
        print(j)
        if j == my_shape[1] - 1:
            x = data[j][0][:, 0]
            y = data[j][0][:, 1]
            z = data[j][0][:, 2]
            dx = resids[0][j]
            dy = resids[1][j]
            dz = resids[2][j]
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(x, y, data[j][0][:, 5], marker='.')
            ax.scatter(x, y, dz, marker='.')

            ax.set_xlabel('$x$', fontsize='10')
            ax.set_ylabel('$y$', fontsize='10')
            ax.set_zlabel('$\Delta z$ residual', fontsize='10')
            plt.show()
            # break
        # break
    # break