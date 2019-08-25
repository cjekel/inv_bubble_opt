import numpy as np
import matplotlib.pyplot as plt
import os

homeuser = os.path.expanduser('~')

filenames = ['residuals_blue00.npy',
             'residuals_blue01.npy',
             'residuals_blue02.npy',
             'residuals_blue03.npy']
datanames = ['blue00.npy',
             'blue01_rotated_90.npy',
             'blue02_rotated_90.npy',
             'blue03.npy']

for i in range(0, 4):
# for residual_file in filenames:
    resids = np.load(filenames[i], allow_pickle=True)
    print(resids.shape)
    data = np.load(os.path.join(homeuser, datanames[i]), allow_pickle=True)
    plt.figure()

    for j, _ in enumerate(resids):
        x = data[i][0][:, 0]
        y = data[i][0][:, 1]
        z = data[i][0][:, 2]
        dx = resids[0][j, :]
        plt.plot(x, dx, 'xk')
    plt.show()
    break