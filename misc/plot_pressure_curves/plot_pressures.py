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
import os
import matplotlib.pyplot as plt

# load the test data
homeuser = os.path.expanduser('~')
blue00 = np.load(os.path.join(homeuser, 'blue00.npy'), allow_pickle=True)
blue01 = np.load(os.path.join(homeuser, 'blue01_rotated_90.npy'),
                 allow_pickle=True)
blue02 = np.load(os.path.join(homeuser, 'blue02_rotated_90.npy'),
                 allow_pickle=True)
blue03 = np.load(os.path.join(homeuser, 'blue03.npy'), allow_pickle=True)

time00 = np.array([2.280572, 2.373457, 2.559227, 2.652112, 2.930767, 3.116537,
                   3.302307, 3.395192, 3.673847, 3.766732, 3.952502, 4.231157,
                   4.509812, 4.788467, 5.067122, 5.160007, 5.345777, 5.531547,
                   5.717317, 5.810202, 6.088857, 6.181742, 6.553282, 6.739052,
                   6.924822, 7.110592, 7.203477, 7.575017, 7.760787, 7.946557,
                   8.410982, 8.503867, 8.782522, 8.968292, 9.246947, 9.432717,
                   9.618487, 9.897142, 9.990027, 10.082912, 10.268682,
                   10.361567, 10.547337, 10.733107, 10.918877, 11.197532,
                   11.290417, 11.383302, 11.476187])
time00 = time00 - time00[0]
time01 = np.array([0.944677, 1.250677, 1.352677, 1.658677, 1.760677, 1.862677,
                   2.066677])
time01 = time01 - time01[0]
time02 = np.array([2.440407, 2.542510, 2.746716, 2.950922, 3.155128, 3.257231,
                   3.359334, 3.563540, 3.767745, 3.971951, 4.176157, 4.380363,
                   4.584569, 4.992981, 5.095085, 5.197187, 5.401393, 5.503497,
                   5.605599, 5.809805, 5.911909, 6.116115, 6.320321, 6.524527,
                   6.728733, 6.932939, 7.137145, 7.341351, 7.647659, 7.749763,
                   7.953969, 8.158175, 8.362380, 8.668690, 8.872896, 9.179204])
time02 = time02 - time02[0]
plt.figure()
plt.plot(time00, blue00[:, 1]*10000, '-', label='Test 1')
plt.plot(time01, blue01[:, 1]*10000, '--', label='Test 2')
plt.plot(time02, blue02[:, 1]*10000, '-.', label='Test 3')
time03 = np.linspace(0.0, 2.2, 11)
plt.plot(time03, blue03[:, 1][18:]*10000, ':', label='Test 4')
plt.legend()
plt.xlabel('Time, seconds')
plt.ylabel('Pressure, bar')
plt.savefig('inflation_pressure.pdf', dpi=300, bbox_inches='tight')
plt.show()