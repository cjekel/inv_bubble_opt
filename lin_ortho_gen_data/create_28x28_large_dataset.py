import joblib
import h5py
import numpy as np
# import matplotlib.pyplot as plt
import invbubble
from sklearn.preprocessing import MinMaxScaler

# load h5 data :)
filenames = ['data2k.hdf5', 'data4k.hdf5', 'data6k.hdf5', 'data8k.hdf5',
             'data10k.hdf5']
n_data = 2000


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

new_dataset = np.zeros((5*n_data, n_c*n_c, 3), dtype=np.single)
for j in range(5):
    data = load_data(filenames[j])
    print('loading data')
    for i in range(n_data):
        # fit rbf and evaulte
        dxh = invbubble.rbf_function(xy_data[0, :, :2], data[i, -1, :, 0], xyh)
        dyh = invbubble.rbf_function(xy_data[0, :, :2], data[i, -1, :, 1], xyh)
        dzh = invbubble.rbf_function(xy_data[0, :, :2], data[i, -1, :, 2], xyh)
        # set outside bounds to 0.0
        dxh[inds] = 0.
        dyh[inds] = 0.
        dzh[inds] = 0.
        new_dataset[(j*n_data) + i, :, 0] = dxh
        new_dataset[(j*n_data) + i, :, 1] = dyh
        new_dataset[(j*n_data) + i, :, 2] = dzh
        # print('i:', i, dxh.max(), dyh.max(), dzh.max())

# print the max for each direction
print('dx max:', new_dataset[:, :, 0].max())
print('dy max:', new_dataset[:, :, 1].max())
print('dz max:', new_dataset[:, :, 2].max())


# reshape the dataset for 0,1 minmax transfer
new_dataset = new_dataset.reshape((5*n_data*n_c*n_c, 3))
scaler = MinMaxScaler()

new_dataset = scaler.fit_transform(new_dataset)
# save the joblib transformer
joblib.dump(scaler, 'minmaxscaler_28x28_10k.z')

# reshape the dataset back to images
new_dataset = new_dataset.reshape((5*n_data, n_c, n_c, 3))

# save the data
# np.save('10k28x28data.npy', new_dataset)
with h5py.File('10k28x28data.hdf5', 'w') as f: 
    dset = f.create_dataset('10k28x28data', data=new_dataset,
                            compression="gzip") 