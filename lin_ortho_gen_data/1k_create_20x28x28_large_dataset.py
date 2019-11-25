import joblib
import h5py
import numpy as np
# import matplotlib.pyplot as plt
import invbubble
from sklearn.preprocessing import MinMaxScaler

# load h5 data :)
filenames = ['data_material_1k.hdf5']
n_data = 477
n_loads = 20


def load_data(filename):
    with h5py.File(filename, 'r') as f:
        # List all groups
        print("Keys: %s" % f.keys())
        a_group_key = list(f.keys())[0]

        data = f[a_group_key][()]
    return data


xy_data = np.load('xy_model.npy')
x = xy_data[0, :, 0]
y = xy_data[0, :, 1]
xy = xy_data[0, :, :2]

# new xy data to generate predictions
n_c = 28
xh = np.linspace(-100, 100, n_c)
yh = np.linspace(-100, 100, n_c)
xh, yh = np.meshgrid(xh, yh)
xyh = np.zeros((n_c*n_c, 2))
xyh[:, 0] = xh.flatten()
xyh[:, 1] = yh.flatten()

inds = np.argwhere(xyh[:, 0]**2 + xyh[:, 1]**2 > 100.**2)

load = np.linspace(0.001, 3.0, 200)
n_loadz = 26
n_loads = 20
pressure_ind_list = np.array([(i+1)*200//n_loadz - 2 for i in range(n_loadz)])
pressures = load[pressure_ind_list + 1]
pressure_ind_list = pressure_ind_list[:20]
pressures = pressures[:20]

new_dataset = np.zeros((n_data, n_loads, n_c*n_c, 3), dtype=np.single)
for j in range(1):
    data = load_data(filenames[j])
    print('loading data')
    for i in range(n_data):
        for kind, k in enumerate(pressure_ind_list):
            # fit rbf and evaulte
            dxh = invbubble.rbf_function(xy_data[0, :, :2], data[i, k, :, 0],
                                         xyh)
            dyh = invbubble.rbf_function(xy_data[0, :, :2], data[i, k, :, 1],
                                         xyh)
            dzh = invbubble.rbf_function(xy_data[0, :, :2], data[i, k, :, 2],
                                         xyh)
            # set outside bounds to 0.0
            dxh[inds] = 0.
            dyh[inds] = 0.
            dzh[inds] = 0.
            new_dataset[(j*n_data) + i, kind, :, 0] = dxh
            new_dataset[(j*n_data) + i, kind, :, 1] = dyh
            new_dataset[(j*n_data) + i, kind, :, 2] = dzh
        print('i:', i, dxh.max(), dyh.max(), dzh.max())


# print the max for each direction
print('dx max:', new_dataset[:, :, 0].max())
print('dy max:', new_dataset[:, :, 1].max())
print('dz max:', new_dataset[:, :, 2].max())


# reshape the dataset for 0,1 minmax transfer
new_dataset = new_dataset.reshape((n_data*n_loads*n_c*n_c, 3))
scaler = MinMaxScaler()

new_dataset = scaler.fit_transform(new_dataset)
# save the joblib transformer
joblib.dump(scaler, 'minmaxscaler_20x28x28_1k.z')

# reshape the dataset back to images
new_dataset = new_dataset.reshape((n_data, n_loads, n_c, n_c, 3))

# save the data
with h5py.File('1k20x28x28data.hdf5', 'w') as f:
    dset = f.create_dataset('1k20x28x28data', data=new_dataset,
                            compression="gzip")
