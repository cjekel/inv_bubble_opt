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
    test_data = [blue03]

    # initialize a maximum objective value
    max_obj = 30.0  # mm

    opt_hist_file = 'ind04r00.csv'
    header = ['E', 'G', 'OBJ', 'Success']
    my_opt = invbubble.BubbleOpt(opt_hist_file, header, max_obj,
                                 None, None,
                                 test_data=test_data,
                                 weights=[1.0, 1.0, 0.103],
                                 mat_model='iso-two')

    np.random.seed(121)

    my_bounds = np.zeros((2, 2))
    my_bounds[0, 0] = 0.12
    my_bounds[0, 1] = 0.25
    my_bounds[1, 0] = 0.2
    my_bounds[1, 1] = 0.9

    X = np.array([[0.166, 0.60],
                  [0.155, 0.52],
                  [0.193, 0.67],
                  [0.167, 0.56],
                  [0.198, 0.7]])

    xres = np.zeros_like(X)
    fres = np.zeros(5)
    for i, x0 in enumerate(X):
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
    invbubble.send_email('cjekel@ufl.edu', 'ind blue 04 done', message)
