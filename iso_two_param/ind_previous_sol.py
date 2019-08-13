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
from scipy.optimize import fmin_l_bfgs_b


def compute_g(E):
    g = E/(2.0*1.24)
    return g*10.0


if __name__ == "__main__":
    invbubble.delete_files()

    # load the test data
    homeuser = os.path.expanduser('~')
    blue00 = np.load(os.path.join(homeuser, 'blue00.npy'),
                     allow_pickle=True)
    blue01 = np.load(os.path.join(homeuser, 'blue01_rotated_90.npy'),
                     allow_pickle=True)
    blue02 = np.load(os.path.join(homeuser, 'blue02_rotated_90.npy'),
                     allow_pickle=True)
    blue03 = np.load(os.path.join(homeuser, 'blue03.npy'),
                     allow_pickle=True)
    tests = [[blue00], [blue01], [blue02], [blue03]]

    # initialize a maximum objective value
    max_obj = 30.0  # mm
    
    header = ['E', 'G', 'OBJ', 'Success']

    np.random.seed(121)

    my_bounds = np.zeros((2, 2))
    my_bounds[0, 0] = 0.12
    my_bounds[0, 1] = 0.32
    my_bounds[1, 0] = 0.1
    my_bounds[1, 1] = 2.0

    X = np.array([[0.2832392, 0.0],
                  [0.29158096, 0.0],
                  [0.28168442, 0.0],
                  [0.2543283, 0.0]])
    for i, x0 in enumerate(X):
        X[i, 1] = compute_g(x0[0])
    xres = np.zeros_like(X)
    fres = np.zeros(4)
    for i, x0 in enumerate(X):
        opt_hist_file = 'indprev' + str(i) + '.csv'
        my_opt = invbubble.BubbleOpt(opt_hist_file, header, max_obj,
                                     None, None,
                                     test_data=tests[i],
                                     weights=[1.0, 1.0, 0.103],
                                     mat_model='iso-two')
        res = fmin_l_bfgs_b(my_opt.calc_obj_function_test_data, x0,
                            approx_grad=True, bounds=my_bounds, factr=1e12,
                            pgtol=1e-06, epsilon=1e-3, iprint=1, m=10000,
                            maxfun=200, maxiter=10, maxls=20)
        xres[i] = res[0]
        fres[i] = res[1]

    # find the best result
    best_ind = np.argmin(fres)
    message = '\nBest result: \n' + str(fres[best_ind]) + """\n
               Best values: \n""" + str(xres[best_ind]) + """\n
               The full result: \n""" + str(fres) + """\n
               Full values: \n""" + str(xres)
    print(message)
    invbubble.send_email('cjekel@ufl.edu', 'ind blue 01 done', message)
