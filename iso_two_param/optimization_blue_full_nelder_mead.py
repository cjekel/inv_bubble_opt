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
from os.path import expanduser
from GPyOpt.methods import BayesianOptimization
# from scipy.optimize import fmin_l_bfgs_b
# from scipy.optimize import minimize
import constrNMPy as cNM 


if __name__ == "__main__":
    invbubble.delete_files()

    # load the test data
    blue00 = np.load(expanduser('~blue00.npy'))
    blue01 = np.load(expanduser('~blue01.npy'))
    blue02 = np.load(expanduser('~blue02.npy'))
    blue03 = np.load(expanduser('~blue03.npy'))
    test_data = [blue00, blue01, blue02, blue03]

    # initialize a maximum objective value
    max_obj = 30.0  # mm

    opt_hist_file = 'my_blue_history_fullnm.csv'
    header = ['E1', 'nu12', 'OBJ', 'Success']
    my_opt = invbubble.BubbleOpt(opt_hist_file, header, max_obj,
                                 None, None,
                                 test_data=test_data,
                                 mat_model='iso-two')

    def conv_my_obj(x):
        f = np.zeros(x.shape[0])
        for i, j in enumerate(x):
            f[i] = my_opt.calc_obj_function_test_data(j)
        return f

    bounds = [{'name': 'var_1', 'type': 'continuous', 'domain': [0.1, 2.0]},
              {'name': 'var_2', 'type': 'continuous', 'domain': [0.01, 0.49]}]

    max_iter = 20
    np.random.seed(121)
    # myBopt = BayesianOptimization(conv_my_obj, domain=bounds, model_type='GP',
    #                               initial_design_numdata=10,
    #                               initial_design_type='latin',
    #                               exact_feval=True, verbosity=True,
    #                               verbosity_model=False)

    # myBopt.run_optimization(max_iter=max_iter, eps=1e-7, verbosity=True,
    #                         report_file='gp_opt_results')

    # print('\n \n EGO Opt Complete \n')
    # print('X values:', myBopt.x_opt)
    # print('Function value:', myBopt.fx_opt)

    my_bounds = np.zeros((2, 2))
    my_bounds[0, 0] = 0.1
    my_bounds[0, 1] = 2.0
    my_bounds[1, 0] = 0.01
    my_bounds[1, 1] = 0.49
    x0 = [0.16874658, 0.49]

    # res = fmin_l_bfgs_b(my_opt.calc_obj_function_test_data, x0,
    #                     approx_grad=True, bounds=my_bounds, factr=10.,
    #                     pgtol=1e-06, epsilon=1e-2, iprint=1,
    #                     maxfun=120, maxiter=120, maxls=20)
    # res = minimize(my_opt.calc_obj_function_test_data, x0, bounds=my_bounds,
    #                method='Nelder-Mead', tol=None, callback=None,
    #                options={'maxfev': 100, 'disp': True, 'return_all': True,
    #                'xatol': 1e-4, 'fatol': 1e-6, 'adaptive': False})
    res = cNM.constrNM(my_opt.calc_obj_function_test_data, x0,
                       my_bounds[:, 0], my_bounds[:, 1], xtol=1e-4, ftol=1e-6,
                       maxfun=100, disp=True, full_output=True)
    print(res)
