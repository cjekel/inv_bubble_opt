import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import Rbf
import invbubble
from scipy.interpolate import interp1d
from sklearn.preprocessing import MinMaxScaler
homeuser = os.path.expanduser('~')

load = np.linspace(0.001, 3.0, 200)
n_loadz = 26
n_loads = 20
pressure_ind_list = np.array([(i+1)*200//n_loadz - 2 for i in range(n_loadz)])
pressures = load[pressure_ind_list + 1]
pressures = pressures[:20]

datanames = ['blue00.npy',
             'blue01_rotated_90.npy',
             'blue02_rotated_90.npy',
             'blue03.npy']

blue00 = np.load(os.path.join(homeuser, datanames[0]),
                 allow_pickle=True)
blue01 = np.load(os.path.join(homeuser, datanames[1]),
                 allow_pickle=True)
blue02 = np.load(os.path.join(homeuser, datanames[2]),
                 allow_pickle=True)
blue03 = np.load(os.path.join(homeuser, datanames[3]),
                 allow_pickle=True)
# plt.figure()
# plt.plot(blue00[:, 1]*1e4, np.ones(blue00[:, 1].size), '.')
# plt.plot(blue01[:, 1]*1e4, 2*np.ones(blue01[:, 1].size), '.')
# plt.plot(blue02[:, 1]*1e4, 3*np.ones(blue02[:, 1].size), '.')
# plt.plot(blue03[:, 1]*1e4, 4*np.ones(blue03[:, 1].size), '.')


# plt.plot(pressures, np.zeros_like(pressures), 'ok')

# plt.show()

# new xy data to generate predictions
n_c = 28
xh = np.linspace(-100, 100, n_c)
yh = np.linspace(-100, 100, n_c)

spacing = xh[0] - xh[1]
xh, yh = np.meshgrid(xh, yh)
inds = np.argwhere(xh[:, 0]**2 + yh[:, 1]**2 > 100.**2)

xyh = np.zeros((n_c*n_c, 2))
xyh[:, 0] = xh.flatten()
xyh[:, 1] = yh.flatten()

test_data_images = []
interplated_test_images = []
for k in range(len(datanames)):
    blue00 = np.load(os.path.join(homeuser, datanames[k]),
                     allow_pickle=True)
    n_p = blue00.shape[0]
    new_dataset = np.zeros((n_p, n_c, n_c, 3), dtype=np.single)

    for p in range(n_p):
        data = blue00[p, 0]
        x = data[:, 0]
        y = data[:, 1]
        for i in range(n_c):
            for j in range(n_c):
                if xh[i, j]**2 + yh[i, j]**2 <= 100.**2:
                    xyinds = np.where((x > xh[i, j]) &
                                      (x < xh[i, j] - spacing) &
                                      (y > yh[i, j]) &
                                      (y < yh[i, j] - spacing))
                    if xyinds[0].size > 0:
                        new_dataset[p, i, j, 0] = data[xyinds, 3].mean()
                        new_dataset[p, i, j, 1] = data[xyinds, 4].mean()
                        new_dataset[p, i, j, 2] = data[xyinds, 5].mean()

    test_data_images.append(new_dataset)

    image_dataset = np.zeros((n_loads, n_c, n_c, 3))

    fdx = interp1d(blue00[:, 1]*1e4, new_dataset[:, :, :, 0], kind='linear',
                   axis=0, copy=True, bounds_error=True, fill_value=0.0,
                   assume_sorted=True)
    fdy = interp1d(blue00[:, 1]*1e4, new_dataset[:, :, :, 1], kind='linear',
                   axis=0, copy=True, bounds_error=True, fill_value=0.0,
                   assume_sorted=True)
    fdz = interp1d(blue00[:, 1]*1e4, new_dataset[:, :, :, 2], kind='linear',
                   axis=0, copy=True, bounds_error=True, fill_value=0.0,
                   assume_sorted=True)
    image_dataset[:, :, :, 0] = fdx(pressures)
    image_dataset[:, :, :, 1] = fdy(pressures)
    image_dataset[:, :, :, 2] = fdz(pressures)
    interplated_test_images.append(image_dataset)

for i in range(4):
    data = interplated_test_images[i]
    # save interpolated test data
    np.save('test_data/test' + str(i) + '.npy', data)
    scaler = MinMaxScaler()
    data = data.reshape((n_loads*n_c*n_c, 3))
    data = scaler.fit_transform(data)
    data = data.reshape((n_loads, n_c, n_c, 3))
    for j in range(n_loads):
        plt.figure()
        plt.title('Test' + str(i+1) +
                  ' Displacement x, Pressure: %8.2f bar' % pressures[j])
        plt.imshow(data[j, :, :, 0])
        plt.savefig('test_data/dispx/test_' + str(i) + '_' + str(j) + '.png',
                    bbox_inches='tight')
        plt.figure()
        plt.title('Test' + str(i+1) +
                  ' Displacement y, Pressure: %8.2f bar' % pressures[j])
        plt.imshow(data[j, :, :, 1])
        plt.savefig('test_data/dispy/test_' + str(i) + '_' + str(j) + '.png',
                    bbox_inches='tight')

        plt.figure()
        plt.title('Test' + str(i+1) +
                  ' Displacement z, Pressure: %8.2f bar' % pressures[j])
        plt.imshow(data[j, :, :, 2])
        plt.savefig('test_data/dispz/test_' + str(i) + '_' + str(j) + '.png',
                    bbox_inches='tight')
        print(data[j, :, :, 0].max(), data[j, :, :, 1].max(), data[j, :, :, 2].max())


# for i in test_data_images:
    # new_dataset = np.zeros((n_loads, n_c, n_c, 3), dtype=np.single)
    # f = interp1d(x, y, kind='linear', axis=-1, copy=True, bounds_error=True, fill_value=0.0, assume_sorted=True)
    # plt.figure()
    # plt.title('Displacement x')
    # plt.imshow(i[-1, :, :, 0])
    # plt.figure()
    # plt.title('Displacement y')
    # plt.imshow(i[-1, :, :, 1])
    # plt.figure()
    # plt.title('Displacement z')
    # plt.imshow(i[-1, :, :, 2])
# # pressures
# pz = np.concatenate((blue00[:, 1], blue01[:, 1], blue02[:, 1], blue03[:, 1]))
# pz.sort()

# plt.figure()
# plt.plot(pz*1e4, np.ones(pz.size), '.')
# plt.show()
