import pyDOE
import numpy as np

my_bounds = np.zeros((3, 2))
my_bounds[0, 0] = 2.0
my_bounds[0, 1] = 3.0
my_bounds[1, 0] = 1.0
my_bounds[1, 1] = 2.0
my_bounds[2, 0] = 1.0
my_bounds[2, 1] = 5.0

n = 10000
np.random.seed(122341)
lhd = pyDOE.lhs(3, samples=n, iterations=10000)
lhd = (my_bounds[:, 1] - my_bounds[:, 0]) * lhd + my_bounds[:, 0]
np.save('mydoe.npy', lhd)
