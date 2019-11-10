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
    x_cv01 = [0.279]
    x_cv02 = [0.222]
    x_cv03 = [0.242]
    x_cv04 = [0.218]
    # weights = [1.0, 1.0, 0.103]
    # x_cv01 = [0.283]
    # x_cv02 = [0.292]
    # x_cv03 = [0.282]
    # x_cv04 = [0.253]
    x = [x_cv01, x_cv02, x_cv03, x_cv04]

    header = ['E1', 'OBJ', 'Success']

    cv01 = invbubble.BubbleOpt('my_full_cv01.csv', header,
                               100.0, None, None,
                               test_data=test_data_cv01,
                               mat_model='iso-one',
                               weights=weights)
    cv02 = invbubble.BubbleOpt('my_full_cv02.csv', header,
                               100.0, None, None,
                               test_data=test_data_cv02,
                               mat_model='iso-one',
                               weights=weights)
    cv03 = invbubble.BubbleOpt('my_full_cv03.csv', header,
                               100.0, None, None,
                               test_data=test_data_cv03,
                               mat_model='iso-one',
                               weights=weights)
    cv04 = invbubble.BubbleOpt('my_full_cv04.csv', header,
                               100.0, None, None,
                               test_data=test_data_cv04,
                               mat_model='iso-one',
                               weights=weights)
    cvs = [cv01, cv02, cv03, cv04]
    fn = ['disp_values_0.npy', 'disp_values_1.npy', 'disp_values_2.npy',
          'disp_values_3.npy']
    for i in range(4):
        _ = cvs[i].calc_obj_function_test_data(x[i], run_abq=True)
        invbubble.read_csv_files(save=True, fn=fn[i])
