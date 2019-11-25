import numpy as np
import pandas as pd
import h5py
from time import time

start_point = 0
end_point = 2000


def read_csv_files(i):
    # initiate array of zeros
    node_values = np.zeros((199, 937, 3), dtype=np.single)
    suc = True
    for j in range(1, 200):
        file_name = 'material_model_data/BubbleTest' + str(i).zfill(5) + '/L00' + str(j).zfill(3) + '.csv'  # noqa E501
        try:
            temp = pd.read_csv(file_name, delimiter=',')
            # grab the x values
            node_values[j - 1, :, 0] = temp.values[:, 11]
            # grab the y values
            node_values[j - 1, :, 1] = temp.values[:, 12]
            # grab the z values
            node_values[j - 1, :, 2] = temp.values[:, 13]
        except FileNotFoundError:
            suc = False

    return node_values, suc


doe = np.load('mydoe_surrogate.npy')
values = []
datas = []

for i, value in enumerate(doe):
    t0 = time()
    data, suc = read_csv_files(i)
    if suc:
        values.append(value)
        datas.append(data)
    t1 = time()
    print('Runtime:', t1-t0, ' i:', i)

values = np.array(values, dtype=np.single)
datas = np.array(datas, dtype=np.single)
np.save('my_success_runs_doe.npy', values)

with h5py.File('data_material_1k.hdf5', 'w') as f:
    dset = f.create_dataset('data_material_1k',
                            data=datas,
                            compression="gzip")
