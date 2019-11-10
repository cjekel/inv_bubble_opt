import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import Rbf

homeuser = os.path.expanduser('~')


def export_data_points(data):
    x_max = 56.25
    y_max = 62.5
    x_zero = 0.0
    y_zero = 0.0
    tol = 0.2
    dx_data = []
    dy_data = []
    dz_data = []
    for d in data:
        # remove the data points not near the area
        x = d[0][:, 0]
        y = d[0][:, 1]
        dx = d[0][:, 3]
        dy = d[0][:, 4]
        dz = d[0][:, 5]

        inds_x_max = np.argwhere(np.abs(x_max - x) <= tol)
        inds_y_max = np.argwhere(np.abs(y_max - y) <= tol)
        inds_x_zero = np.argwhere(np.abs(x_zero - x) <= tol)
        inds_y_zero = np.argwhere(np.abs(y_zero - y) <= tol)
        inds = np.concatenate((inds_x_max, inds_x_zero, inds_y_max,
                               inds_y_zero))

        f_dx = Rbf(x[inds], y[inds], dx[inds], function='linear', smooth=0.1)
        f_dy = Rbf(x[inds], y[inds], dy[inds], function='linear', smooth=0.1)
        f_dz = Rbf(x[inds], y[inds], dz[inds], function='linear', smooth=0.1)

        # generate predictions
        dx_data.append(f_dx(x_max, y_zero))
        dy_data.append(f_dy(x_zero, y_max))
        dz_data.append(f_dz(x_zero, y_zero))
    return np.array(dx_data), np.array(dy_data), np.array(dz_data)


datanames = ['blue00.npy',
             'blue01_rotated_90.npy',
             'blue02_rotated_90.npy',
             'blue03.npy']

for i in range(4):
    data = np.load(os.path.join(homeuser, datanames[i]), allow_pickle=True)
    dx_data, dy_data, dz_data = export_data_points(data)
    np.save('dx_data_' + datanames[i], dx_data)
    np.save('dy_data_' + datanames[i], dy_data)
    np.save('dz_data_' + datanames[i], dz_data)

    plt.figure()
    plt.plot(np.arange(len(dx_data)), dx_data, label='dx')
    plt.plot(np.arange(len(dy_data)), dy_data, label='dy')
    plt.plot(np.arange(len(dz_data)), dz_data, label='dz')
    plt.legend()
plt.show()
