import numpy as np
import invbubble
from time import time
from shutil import copytree

start_point = 2000
end_point = 2010

# load_data
Y = np.load('../mydoe.npy')


def run_abq_model(x):
    val = 1

    invbubble.write_material_model(x)
    # run the finite element model
    val = invbubble.run_model()

    if val == 0:
        # check the status file to ensure the FE model was successful
        suc = invbubble.read_sta()
    else:
        suc = False
    return val, suc


for i in range(start_point, end_point+1):
    t0 = time()
    val, suc = run_abq_model(Y[i])
    if suc is True:
        if val == 0:
            val2 = invbubble.export_csv_files()
            if val2 == 0:
                # copy files to new dir
                copytree('BubbleTest', 'BubbleTest' + str(i).zfill(5))
    invbubble.delete_files()
    t1 = time()
    print('Runetime:', t1 - t0, ' i:', i)
