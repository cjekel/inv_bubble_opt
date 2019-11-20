import numpy as np
import pandas as pd
import h5py
from time import time

start_point = 0
end_point = 2000


def read_csv_files(i):
    # initiate array of zeros
    node_values = np.zeros((199, 937, 3), dtype=np.single)

    for i in range(1, 200):
        file_name = 'BubbleTest' + str(i).zfill(5) + '/L00' + str(i).zfill(3) + '.csv'  # noqa E501
        temp = pd.read_csv(file_name, delimiter=',')
        # grab the x values
        node_values[i - 1, :, 0] = temp.values[:, 11]
        # grab the y values
        node_values[i - 1, :, 1] = temp.values[:, 12]
        # grab the z values
        node_values[i - 1, :, 2] = temp.values[:, 13]

    return node_values


data = np.zeros((end_point-start_point, 199, 937, 3), dtype=np.single)

for i in range(start_point, end_point):
    t0 = time()
    data[i] = read_csv_files(i)
    t1 = time()
    print('Runtime:', t1-t0)

with h5py.File('data2k.hdf5', 'w') as f:
    dset = f.create_dataset('data2k', data=data, compression="gzip")
