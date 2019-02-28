# MIT License

# Copyright (c) 2019 Charles Jekel

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import numpy as np
import invbubble
import os
from GPyOpt.methods import BayesianOptimization
from scipy.optimize import fmin_l_bfgs_b


if __name__ == "__main__":
    invbubble.delete_files()

    # load the test data
    homeuser = os.path.expanduser('~')
    blue00 = np.load(os.path.join(homeuser, 'blue00.npy'))
    blue01 = np.load(os.path.join(homeuser, 'blue01.npy'))
    blue02 = np.load(os.path.join(homeuser, 'blue02.npy'))
    blue03 = np.load(os.path.join(homeuser, 'blue03.npy'))
    test_data = [blue00, blue01, blue02, blue03]

    # initialize a maximum objective value
    max_obj = 30.0  # mm

    opt_hist_file = 'g12stability.csv'
    header = ['E1', 'E2', 'G12', 'OBJ', 'Success']
    my_opt = invbubble.BubbleOpt(opt_hist_file, header, max_obj,
                                 None, None,
                                 test_data=test_data)
    E1 = 0.26
    E2 = 0.24
    G12spread = np.logspace(-1, 1.5, num=50)
    for G12 in G12spread:
        my_opt.calc_obj_function_test_data([E1, E2, G12])
