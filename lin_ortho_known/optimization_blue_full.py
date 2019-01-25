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
import pandas as pd
import invbubble
from GPyOpt.methods import BayesianOptimization
from scipy.optimize import fmin_l_bfgs_b


if __name__ == "__main__":
    # Known material model: x = [800.0*1e-3, 150.0*1e-3, 25.0*1e-3]
    # x = [800.0*1e-3, 150.0*1e-3, 25.0*1e-3]
    invbubble.delete_files()
    # load starting points from previous optimization
    prev_res = pd.read_csv('../opt_results/00/my_gp_ei_history.csv')
    max_obj = prev_res.values[:, 4].max()

    opt_hist_file = '~/my_blue_history.csv'
    header = ['E1', 'E2', 'G12', 'OBJ', 'Success']
    my_opt = invbubble.BubbleOpt(opt_hist_file, header, max_obj,
                                 'xy_model.npy', 'disp_values.npy')

    def conv_my_obj(x):
        f = np.zeros(x.shape[0])
        for i, j in enumerate(x):
            f[i] = my_opt.calc_obj_function_abq_data(j)
        return f

    bounds = [{'name': 'var_1', 'type': 'continuous', 'domain': [0.1, 2.0]},
              {'name': 'var_2', 'type': 'continuous', 'domain': [0.05, 1.0]},
              {'name': 'var_3', 'type': 'continuous', 'domain': [0.01, 0.2]}]

    max_iter = 30
    np.random.seed(121)
    myBopt = BayesianOptimization(conv_my_obj, domain=bounds, model_type='GP',
                                  initial_design_numdata=20,
                                  initial_design_type='latin',
                                  exact_feval=True, verbosity=True,
                                  verbosity_model=False)

    myBopt.run_optimization(max_iter=max_iter, eps=1e-7, verbosity=True,
                            report_file='gp_opt_results')

    print('\n \n EGO Opt Complete \n')
    print('X values:', myBopt.x_opt)
    print('Function value:', myBopt.fx_opt)

    my_bounds = np.zeros((3, 2))
    my_bounds[0, 0] = 0.1
    my_bounds[0, 1] = 2.0
    my_bounds[1, 0] = 0.05
    my_bounds[1, 1] = 1.0
    my_bounds[2, 0] = 0.01
    my_bounds[2, 1] = 0.2

    res = fmin_l_bfgs_b(my_opt.calc_obj_function_abq_data, myBopt.x_opt,
                        approx_grad=True, bounds=my_bounds, factr=1e7,
                        pgtol=1e-06, epsilon=1e-5, iprint=1,
                        maxfun=200, maxiter=120, maxls=25)
    print(res)
