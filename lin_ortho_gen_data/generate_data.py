import numpy as np
import invbubble
import pyDOE
from time import time

my_bounds = np.zeros((3, 2))
my_bounds[0, 0] = 2.0
my_bounds[0, 1] = 3.0
my_bounds[1, 0] = 1.0
my_bounds[1, 1] = 2.0
my_bounds[2, 0] = 1.0
my_bounds[2, 1] = 5.0


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


t0 = time()
a, b = run_abq_model(my_bounds[:, 0])
t1 = time()
print('Runetime:', t1 - t0)
t0 = time()
c, d = run_abq_model(my_bounds[:, 1])
t1 = time()
print('Runetime:', t1 - t0)
