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
    test_data_full = [blue00, blue01, blue02, blue03]
    test_data_cv01 = [blue00]
    test_data_cv02 = [blue01]
    test_data_cv03 = [blue02]
    test_data_cv04 = [blue03]

    # material model results
    # weights = [1.0, 1.0, 1.0]
    # x_cv01 = [0.21438219, 0.24411916, 0.35898253]
    # x_cv02 = [0.2915786,  0.24324141, 0.36537261]
    # x_cv03 = [0.30254224, 0.22681066, 0.47303238]
    # x_cv04 = [0.28004711, 0.24353676, 0.32121493]
    weights = [1.0, 1.0, 0.103]
    x_cv01 = [0.30623759, 0.23023922, 0.47506611]
    x_cv02 = [0.30407659, 0.2292173, 0.4766851]
    x_cv03 = [0.30426304, 0.22930547, 0.47602973]
    x_cv04 = [0.30461725, 0.22947298, 0.47608423]
    x = [x_cv01, x_cv02, x_cv03, x_cv04]

    header = ['E1', 'E2', 'G12', 'OBJ', 'Success']

    # initialize the bubble objects
    # my_full = invbubble.BubbleOpt('my_full_test.csv', header,
    #                               100.0, None, None,
    #                               test_data=test_data_full,
    #                               mat_model='lin-ortho')
    cv01 = invbubble.BubbleOpt('my_full_cv01.csv', header,
                               100.0, None, None,
                               test_data=test_data_cv01,
                               mat_model='lin-ortho',
                               weights=weights)
    cv02 = invbubble.BubbleOpt('my_full_cv02.csv', header,
                               100.0, None, None,
                               test_data=test_data_cv02,
                               mat_model='lin-ortho',
                               weights=weights)
    cv03 = invbubble.BubbleOpt('my_full_cv03.csv', header,
                               100.0, None, None,
                               test_data=test_data_cv03,
                               mat_model='lin-ortho',
                               weights=weights)
    cv04 = invbubble.BubbleOpt('my_full_cv04.csv', header,
                               100.0, None, None,
                               test_data=test_data_cv04,
                               mat_model='lin-ortho',
                               weights=weights)
    sep = np.zeros((4, 4))
    cv_scores = np.zeros(4)
    cv_ind = np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]])
    cv_values = np.zeros(4)
    for i in range(4):
        sep[i, 0] = cv01.calc_obj_function_test_data(x[i], run_abq=True)
        sep[i, 1] = cv02.calc_obj_function_test_data(x[i], run_abq=False)
        sep[i, 2] = cv03.calc_obj_function_test_data(x[i], run_abq=False)
        sep[i, 3] = cv04.calc_obj_function_test_data(x[i], run_abq=False)
        cv_scores[i] = sep[i, cv_ind[i]].mean()
        # break

    cv_values = sep.diagonal()

    # np.save('blue_cross_compute_res.npy', results)
    np.save('cv_sep.npy', sep)

    # print(results)
    # print('..')
    print(sep)
    print(cv_scores)
    print('CV values: \n', cv_values)
