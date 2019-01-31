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

    # material model results
    x_full = [0.26422968, 0.24658871, 0.00257984*100.]
    x_cv01 = [0.34069393, 0.17930365, 0.00824243*100.]

    e1 = np.linspace(x_full[0], x_cv01[0], 100)
    e2 = np.linspace(x_full[1], x_cv01[1], 100)
    g12 = np.linspace(x_full[2], x_cv01[2], 100)

    header = ['E1', 'E2', 'G12', 'OBJ', 'Success']

    # initialize the bubble objects
    my_full = invbubble.BubbleOpt('my_line_search.csv', header,
                                  100.0, None,
                                  None,
                                  test_data=test_data_full,
                                  mat_model='lin-ortho')

    results = np.zeros(100)

    for i in range(100):
        results[i] = my_full.calc_obj_function_test_data([e1[i], e2[i], g12[i]])

    np.save('line_search_full_cv01.npy', results)

    print(results)
