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

    # set up the various data configurations
    test_data_cv01 = [blue00]
    test_data_cv02 = [blue01]
    test_data_cv03 = [blue02]
    test_data_cv04 = [blue03]

    # material model results
    weights = [1.0, 1.0, 1.0]
    x_cv01 = [0.34281612, 0.24753703, 0.47521649]
    x_cv02 = [0.21217128, 0.24056701, 0.35279843]
    x_cv03 = [0.21734322, 0.25732016, 0.36265037]
    x_cv04 = [0.24255713, 0.21240263, 0.35930347]
    x = [x_cv01, x_cv02, x_cv03, x_cv04]

    header = ['E1', 'E2', 'G12', 'OBJ', 'Success']

    cv01 = invbubble.BubbleOpt('my_full_cv01.csv', header,
                               100.0, None, None,
                               test_data=test_data_cv01,
                               mat_model='lin-ortho',
                               weights=weights,
                               residual_fn='residuals_blue00.npy')
    cv02 = invbubble.BubbleOpt('my_full_cv02.csv', header,
                               100.0, None, None,
                               test_data=test_data_cv02,
                               mat_model='lin-ortho',
                               weights=weights,
                               residual_fn='residuals_blue01.npy')
    cv03 = invbubble.BubbleOpt('my_full_cv03.csv', header,
                               100.0, None, None,
                               test_data=test_data_cv03,
                               mat_model='lin-ortho',
                               weights=weights,
                               residual_fn='residuals_blue02.npy')
    cv04 = invbubble.BubbleOpt('my_full_cv04.csv', header,
                               100.0, None, None,
                               test_data=test_data_cv04,
                               mat_model='lin-ortho',
                               weights=weights,
                               residual_fn='residuals_blue03.npy')
    cvs = [cv01, cv02, cv03, cv04]
    for i in range(4):
        _ = cvs[i].calc_obj_function_test_data(x[i], run_abq=True)
        # sep[i, 2] = cv03.calc_obj_function_test_data(x[i], run_abq=False)
        # sep[i, 3] = cv04.calc_obj_function_test_data(x[i], run_abq=False)
        # cv_scores[i] = sep[i, cv_ind[i]].mean()
        # break

    # cv_values = sep.diagonal()

    # np.save('blue_cross_compute_res.npy', results)
    # np.save('cv_sep.npy', sep)

    # print(results)
    # print('..')
    # print(sep)
    # print(cv_scores)
    # print('CV values: \n', cv_values)