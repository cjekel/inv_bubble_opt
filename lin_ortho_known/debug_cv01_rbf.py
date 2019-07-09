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
    blue00 = np.load(os.path.join(homeuser, 'blue00.npy'), allow_pickle=True)
    blue01 = np.load(os.path.join(homeuser, 'blue01.npy'), allow_pickle=True)
    blue02 = np.load(os.path.join(homeuser, 'blue02.npy'), allow_pickle=True)
    blue03 = np.load(os.path.join(homeuser, 'blue03.npy'), allow_pickle=True)

    # set up the various data configurations
    # test_data_full = [blue00, blue01, blue02, blue03]
    test_data_cv01 = [blue01, blue02, blue03]
    # test_data_cv02 = [blue00, blue02, blue03]
    # test_data_cv03 = [blue00, blue01, blue03]
    # test_data_cv04 = [blue00, blue01, blue02]

    # material model results
    # x_full = [0.31173864, 0.23519048, 0.47272037]
    x_cv01 = [0.22084207, 0.27291883, 0.36580145]
    x_blue01 = [0.31248343, 0.23532769, 0.47470262]
    x_blue02 = [0.29935206, 0.23869944, 0.40985012]
    x_blue03 = [0.27680849, 0.21697615, 0.44515473]
    x = [x_cv01, x_blue01, x_blue02, x_blue03]

    header = ['E1', 'E2', 'G12', 'OBJ', 'Success']
    'my_full_test.csv'

    # initialize the bubble objects
    # my_full = invbubble.BubbleOpt('my_full_test.csv', header,
    #                               100.0, None, None,
    #                               test_data=test_data_full,
    #                               mat_model='lin-ortho',
    #                               weights=[1.0, 1.0, .103],
    #                               debug=True)
    my_cv01 = invbubble.BubbleOpt('my_full_cv01.csv', header,
                                  100.0, None, None,
                                  test_data=test_data_cv01,
                                  mat_model='lin-ortho',
                                  weights=[1.0, 1.0, .103],
                                  debug=True)
    my_cv01d = invbubble.BubbleOpt('my_full_cv01.csv', header,
                                   100.0, None, None,
                                   test_data=test_data_cv01,
                                   mat_model='lin-ortho',
                                   weights=[1.0, 1.0, .103],
                                   debug=True,
                                   MyInt=invbubble.InterpolateThenRBF)
    my_blue01 = invbubble.BubbleOpt('my_full_cv02.csv', header,
                                    100.0, None, None,
                                    test_data=[blue01],
                                    mat_model='lin-ortho',
                                    weights=[1.0, 1.0, .103],
                                    debug=True)
    my_blue02 = invbubble.BubbleOpt('my_full_cv03.csv', header,
                                    100.0, None, None,
                                    test_data=[blue02],
                                    mat_model='lin-ortho',
                                    weights=[1.0, 1.0, .103],
                                    debug=True)
    my_blue03 = invbubble.BubbleOpt('my_full_cv04.csv', header,
                                    100.0, None, None,
                                    test_data=[blue03],
                                    mat_model='lin-ortho',
                                    weights=[1.0, 1.0, .103],
                                    debug=True)
    results = np.zeros((4, 4))

    for i in range(1):
        results[i, 0] = my_cv01.calc_obj_function_test_data(x[i])
        results[i, 1] = my_cv01d.calc_obj_function_test_data(x[i], run_abq=False)  # noqa

        # results[i, 1] = my_blue01.calc_obj_function_test_data(x[i],
        #                                                       run_abq=False)
        # results[i, 2] = my_blue02.calc_obj_function_test_data(x[i],
        #                                                       run_abq=False)
        # results[i, 3] = my_blue03.calc_obj_function_test_data(x[i],
        #                                                       run_abq=False)
        # results[i, 4] = my_cv04.calc_obj_function_test_data(x[i],
        #                                                     run_abq=False)

    # np.save('blue_cross_compute_res.npy', results)

    print(results)
