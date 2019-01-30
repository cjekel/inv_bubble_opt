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


if __name__ == "__main__":
    invbubble.delete_files()

    # load the test data
    blue00 = np.load(expanduser('blue00.npy'))
    blue01 = np.load(expanduser('blue01.npy'))
    blue02 = np.load(expanduser('blue02.npy'))
    blue03 = np.load(expanduser('blue03.npy'))

    # set up the various data configurations
    test_data_full = [blue00, blue01, blue02, blue03]
    test_data_cv01 = [blue01, blue02, blue03]
    test_data_cv02 = [blue00, blue02, blue03]
    test_data_cv03 = [blue00, blue01, blue03]
    test_data_cv04 = [blue00, blue01, blue02]

    # material model results
    x_full = [0.26422968, 0.24658871, 0.00257984]
    x_cv01 = [0.34069393, 0.17930365, 0.00824243]
    x_cv02 = [0.31135523, 0.23734801, 0.00474714]
    x_cv03 = [0.32258707, 0.21461668, 0.00583026]
    x_cv04 = [0.27982852, 0.2445296, 0.00321272]
    x = [x_full, x_cv01, x_cv02, x_cv03, x_cv04]

    header = ['E1', 'E2', 'G12', 'OBJ', 'Success']
    'my_full_test.csv'

    # initialize the bubble objects
    my_full = invbubble.BubbleOpt('my_full_test.csv', header,
                                  100.0, 'xy_model.npy',
                                  'disp_values.npy',
                                  test_data=test_data_full,
                                  mat_model='lin-ortho')
    my_cv01 = invbubble.BubbleOpt('my_full_cv01.csv', header,
                                  100.0, 'xy_model.npy',
                                  'disp_values.npy',
                                  test_data=test_data_cv01,
                                  mat_model='lin-ortho')
    my_cv02 = invbubble.BubbleOpt('my_full_cv02.csv', header,
                                  100.0, 'xy_model.npy',
                                  'disp_values.npy',
                                  test_data=test_data_cv02,
                                  mat_model='lin-ortho')
    my_cv03 = invbubble.BubbleOpt('my_full_cv03.csv', header,
                                  100.0, 'xy_model.npy',
                                  'disp_values.npy',
                                  test_data=test_data_cv03,
                                  mat_model='lin-ortho')
    my_cv04 = invbubble.BubbleOpt('my_full_cv04.csv', header,
                                  100.0, 'xy_model.npy',
                                  'disp_values.npy',
                                  test_data=test_data_cv04,
                                  mat_model='lin-ortho')
    results = np.zeros((5, 5))

    for i in range(5):
        results[i, 0] = my_full.calc_obj_function_test_data(x[i])
        results[i, 1] = my_cv01.calc_obj_function_test_data(x[i],
                                                            run_abq=False)
        results[i, 2] = my_cv02.calc_obj_function_test_data(x[i],
                                                            run_abq=False)
        results[i, 3] = my_cv03.calc_obj_function_test_data(x[i],
                                                            run_abq=False)
        results[i, 4] = my_cv04.calc_obj_function_test_data(x[i],
                                                            run_abq=False)

    np.save('blue_cross_compute_res.npy', results)

    print(results)
