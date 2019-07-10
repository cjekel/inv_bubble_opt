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
from time import time


if __name__ == "__main__":
    invbubble.delete_files()

    # load the test data
    homeuser = os.path.expanduser('~')
    # blue00 = np.load(os.path.join(homeuser, 'blue00.npy'), allow_pickle=True)
    # blue01 = np.load(os.path.join(homeuser, 'blue01.npy'), allow_pickle=True)
    # blue02 = np.load(os.path.join(homeuser, 'blue02.npy'), allow_pickle=True)
    blue03 = np.load(os.path.join(homeuser, 'blue03.npy'), allow_pickle=True)
    blue03r = np.load(os.path.join(homeuser, 'blue03_rotated_90.npy'), allow_pickle=True)

    x = [0.2, 0.25768191, 0.47262295]
    header = ['E1', 'E2', 'G12', 'OBJ', 'Success']
    'my_full_test.csv'


    my = invbubble.BubbleOpt('my_full_cv01.csv', header,
                             100.0, None, None,
                             test_data=[blue03],
                             mat_model='lin-ortho',
                             weights=[1.0, 1.0, .103],
                             debug=True,
                             MyInt=invbubble.InterpolateSimpleRBF)

    myr = invbubble.BubbleOpt('my_full_cv01.csv', header,
                              100.0, None, None,
                              test_data=[blue03r],
                              mat_model='lin-ortho',
                              weights=[1.0, 1.0, .103],
                              debug=True,
                              MyInt=invbubble.InterpolateSimpleRBF)

    results = np.zeros(2)
    results[0] = my.calc_obj_function_abq_data(x)
    results[1] = myr.calc_obj_function_abq_data(x, run_abq=False)
    # np.save('blue_cross_compute_res.npy', results)

    print(results)
