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

test_data = [blue00, blue01, blue02, blue03]
count = 1
for test in test_data:
    plt.figure()
    plt.plot(test[-1, 0][:, 0], test[-1, 0][:, 1], '.k', markersize=0.5)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')
    plt.savefig(str(count) + '_test_final_p.png', dpi=300, bbox_inches='tight')
    count += 1
plt.show()

# plt.figure()
# plt.plot(time00, blue00[:, 1]*10000, '-', label='Test 1')
# plt.plot(time01, blue01[:, 1]*10000, '--', label='Test 2')
# plt.plot(time02, blue02[:, 1]*10000, '-.', label='Test 3')
# time03 = np.linspace(0.0, 2.2, 11)
# plt.plot(time03, blue03[:, 1][18:]*10000, ':', label='Test 4')
# plt.legend()
# plt.xlabel('Time, seconds')
# plt.ylabel('Pressure, bar')
# plt.savefig('inflation_pressure.pdf', dpi=300, bbox_inches='tight')
# plt.show()