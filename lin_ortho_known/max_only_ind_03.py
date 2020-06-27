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
from scipy.optimize import fmin_l_bfgs_b, differential_evolution


if __name__ == "__main__":
    invbubble.delete_files()

    # load the test data
    # load the test data
    homeuser = os.path.expanduser('~')
    blue00 = np.load(os.path.join(homeuser, 'new_max_only_blue00.npy'),
                     allow_pickle=True)
    blue01 = np.load(os.path.join(homeuser, 'new_max_only_blue01_rotated_90.npy'),
                     allow_pickle=True)
    blue02 = np.load(os.path.join(homeuser, 'new_max_only_blue02_rotated_90.npy'),
                     allow_pickle=True)
    blue03 = np.load(os.path.join(homeuser, 'new_max_only_blue03.npy'),
                     allow_pickle=True)
    test_data = [blue02]

    # initialize a maximum objective value
    max_obj = 30.0  # mm

    opt_hist_file = 'ind03r00.csv'
    header = ['E1', 'E2', 'G12', 'OBJ', 'Success']
    my_opt = invbubble.BubbleOpt(opt_hist_file, header, max_obj,
                                 None, None,
                                 test_data=test_data,
                                 weights=[1.0, 1.0, 1.0])

    def conv_my_obj(x):
        f = np.zeros(x.shape[0])
        for i, j in enumerate(x):
            f[i] = my_opt.calc_obj_function_test_data(j)
        return f

    bounds = [{'name': 'var_1', 'type': 'continuous', 'domain': [0.10, 0.4]},
              {'name': 'var_2', 'type': 'continuous', 'domain': [0.05, 0.3]},
              {'name': 'var_3', 'type': 'continuous', 'domain': [0.2, 1.0]}]
    X = np.array([[0.21729489, 0.18,       0.45696582],
                  [0.34281421, 0.24535698, 0.47569612],
                  [0.22663279, 0.18564073, 0.46086685],
                  [0.2,        0.18097237, 0.33507829],
                  [0.3323988,  0.22538752, 0.55021972]])
    # Y = conv_my_obj(X).reshape(-1, 1)
    max_iter = 6
    np.random.seed(121)
    # myBopt = BayesianOptimization(conv_my_obj, domain=bounds,
    #                               model_type='GP',
    #                               X=X, Y=Y,
    #                               initial_design_numdata=0,
    #                               exact_feval=True, verbosity=True,
    #                               verbosity_model=False)

    # myBopt.run_optimization(max_iter=max_iter, eps=1e-7, verbosity=True,
    #                         report_file='gp_opt_results')

    # print('\n \n EGO Opt Complete \n')
    # print('X values:', myBopt.x_opt)
    # print('Function value:', myBopt.fx_opt)

    my_bounds = np.zeros((3, 2))
    my_bounds[0, 0] = 0.1
    my_bounds[0, 1] = 0.4
    my_bounds[1, 0] = 0.05
    my_bounds[1, 1] = 0.3
    my_bounds[2, 0] = 0.2
    my_bounds[2, 1] = 1.0

    # def de_obj(X):
    #     y_hat, _ = myBopt.model.predict(X)
    #     return y_hat

    # print('Minimize differential evolution')
    # res = differential_evolution(de_obj, my_bounds)
    # y_de = my_opt.calc_obj_function_test_data(res.x)
    # if y_de < myBopt.fx_opt:
    #     x0 = res.x
    #     print('Polishing the GP model improved the result')
    # else:
    #     x0 = myBopt.x_opt
    #     print('Polishing the GP model did not help')
    xres = np.zeros_like(X)
    fres = np.zeros(5)
    for i, x0 in enumerate(X):
        res = fmin_l_bfgs_b(my_opt.calc_obj_function_test_data, x0,
                            approx_grad=True, bounds=my_bounds, factr=1e12,
                            pgtol=1e-06, epsilon=1e-2, iprint=1, m=10000,
                            maxfun=200, maxiter=10, maxls=20)
        xres[i] = res[0]
        fres[i] = res[1]
    print(fres)
    print(xres)
    # find the best result
    best_ind = np.argmin(fres)
    message = 'Best result: ' + str(fres[best_ind]) + """\n
               Best values: """ + str(xres[best_ind]) + """\n
               The full result: """ + str(fres) + """\n
               Full values: """ + str(xres)
    print(message)
    invbubble.send_email('cjekel@ufl.edu', 'ind 03 done', message)
