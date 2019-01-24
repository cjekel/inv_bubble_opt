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



if __name__ == "__main__":
    # Known material model: x = [800.0*1e-3, 150.0*1e-3, 25.0*1e-3]
    # x = [800.0*1e-3, 150.0*1e-3, 25.0*1e-3]
    invbubble.delete_files()
    # load starting points from previous optimization
    prev_res = pd.read_csv('../opt_results/00/my_gp_ei_history.csv')
    max_obj = prev_res.values[:, 4].max()

    opt_hist_file = '~/my_gp_ei_history.csv'
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

    max_iter = 360
    np.random.seed(121)
    myBopt = BayesianOptimization(conv_my_obj, domain=bounds, model_type='GP',
                                  initial_design_numdata=0,
                                  initial_design_type='latin',
                                  exact_feval=True, verbosity=True,
                                  verbosity_model=False)

    # asign previous x and y values
    myBopt.X = prev_res.values[:, 1:4]
    myBopt.Y = prev_res.values[:, 4]

    myBopt.run_optimization(max_iter=max_iter, eps=1e-7, verbosity=True,
                            report_file='gp_opt_results')

    print('\n \n Opt found \n')
    print('X values:', myBopt.x_opt)
    print('Function value:', myBopt.fx_opt)
