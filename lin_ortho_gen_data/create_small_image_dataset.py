import joblib
import h5py
import numpy as np
import matplotlib.pyplot as plt
import invbubble
from sklearn.preprocessing import MinMaxScaler

# load h5 data :)
filename = 'data2k.hdf5'
n_data = 10

with h5py.File(filename, 'r') as f:
    # List all groups
    print("Keys: %s" % f.keys())
    a_group_key = list(f.keys())[0]

    data = f[a_group_key][()]

xy_data = np.load('xy_model.npy')
x = xy_data[0, :, 0]
y = xy_data[0, :, 1]
xy = xy_data[0, :, :2]

# new xy data to generate predictions
n_c = 56
xh = np.linspace(-100, 100, n_c)
yh = np.linspace(-100, 100, n_c)
xh, yh = np.meshgrid(xh, yh)
xyh = np.zeros((n_c*n_c, 2))
xyh[:, 0] = xh.flatten()
xyh[:, 1] = yh.flatten()

inds = np.argwhere(xyh[:, 0]**2 + xyh[:, 1]**2 > 100.**2)

new_dataset = np.zeros((n_data, n_c*n_c, 3), dtype=np.single)

for i range(n_data):
    # fit rbf and evaulte
    dxh = invbubble.rbf_function(xy_data[0, :, :2], data[i, -1, :, 0], xyh)
    dyh = invbubble.rbf_function(xy_data[0, :, :2], data[i, -1, :, 1], xyh)
    dzh = invbubble.rbf_function(xy_data[0, :, :2], data[i, -1, :, 2], xyh)
    # set outside bounds to 0.0
    dxh[inds] = 0.
    dyh[inds] = 0.
    dzh[inds] = 0.
    print('i:', i)

# reshape the dataset for 0,1 minmax transfer
new_dataset = new_dataset.reshape((n_data*n_c*n_c, 3))
scaler = MinMaxScaler()
new_dataset = scaler.fit_transform(new_dataset)

# reshape the dataset back to images
new_dataset = new_dataset.reshape((n_data, n_c, n_c, 3))

# save the data
np.save('2k.npy', new_dataset)
