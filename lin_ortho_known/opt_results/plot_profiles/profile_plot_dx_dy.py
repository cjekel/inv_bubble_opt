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
from invbubble import InterpolateSimpleRBF
from scipy.interpolate import interp1d

# load the test data
homeuser = os.path.expanduser('~')
blue00 = np.load(os.path.join(homeuser, 'blue00.npy'), allow_pickle=True)
blue01 = np.load(os.path.join(homeuser, 'blue01_rotated_90.npy'),
                 allow_pickle=True)
blue02 = np.load(os.path.join(homeuser, 'blue02_rotated_90.npy'),
                 allow_pickle=True)
blue03 = np.load(os.path.join(homeuser, 'blue03.npy'), allow_pickle=True)

atol = 2.5e-1
rtol = 2.5e-1
count = 0
n_pred = 1000
X = np.load('../../material_model_results/xy_model.npy')
# labels = ['loc', 'loc_w', 'max', 'max_w']
labels = [r'$e$', r'$e_w$', r'$e_z$']

for i, test in enumerate([blue00, blue01, blue02, blue03]):
    disp_loc = np.load('../../material_model_results/loc_0' + str(i+1) + '.npy')
    disp_loc_w = np.load('../../material_model_results/loc_w_0' + str(i+1) + '.npy')
    disp_max_loc = np.load('../../material_model_results/max_loc_0' + str(i+1) + '.npy')
    disp_max_w_loc = np.load('../../material_model_results/max_loc_w_0' + str(i+1) + '.npy')
    disp_max_z = np.load('../../material_model_results/max_z_0' + str(i+1) + '.npy')

    # set up interpolation models
    int_loc = InterpolateSimpleRBF(X, disp_loc)
    int_loc_w = InterpolateSimpleRBF(X, disp_loc_w)
    int_max_loc = InterpolateSimpleRBF(X, disp_max_loc)
    int_max_w_loc = InterpolateSimpleRBF(X, disp_max_w_loc)
    int_max_z = InterpolateSimpleRBF(X, disp_max_z)

    n = len(test)
    for j in range(n):
        data = test[j][0]
        p = test[j][1]
        logic_1 = np.isclose(data[:, 0], 0.0, atol=atol, rtol=rtol)
        logic_2 = np.isclose(data[:, 1], 0.0, atol=atol, rtol=rtol)
        data_x_0 = data[logic_1]
        data_y_0 = data[logic_2]


        fig = plt.figure(figsize=(10, 6))
        plt.subplot(211)

        # find x_min and x_max index
        x_min_ind = np.argmin(data_y_0[:, 0])
        x_max_ind = np.argmax(data_y_0[:, 0])

        xy = np.zeros((n_pred, 2))
        xy[:, 0] = np.linspace(data_y_0[x_min_ind, 0], data_y_0[x_max_ind, 0], num=n_pred)
        xy[:, 1] = np.linspace(data_y_0[x_min_ind, 1], data_y_0[x_max_ind, 1], num=n_pred)

        # calc values
        dx_loc, dy_loc, dz_loc = int_loc.calc_disp(xy, p)
        dx_loc_w, dy_loc_w, dz_loc_w = int_loc_w.calc_disp(xy, p)
        dx_max_loc, dy_max_loc, dz_max_loc = int_max_loc.calc_disp(xy, p)
        dx_max_loc_w, dy_max_loc_w, dz_max_loc_w = int_max_w_loc.calc_disp(xy, p)
        dx_max_z, dy_max_z, dz_max_z = int_max_z.calc_disp(xy, p)
        plt.title('Test ' + str(count+1) + ", " + str(round(p*1e4, 3)) + " bar", fontsize=18)
        plt.plot(data_y_0[:, 0] + data_y_0[:, 3], data_y_0[:, 3], '.', label='test')
        f = interp1d(data_y_0[:, 0] + data_y_0[:, 3], data_y_0[:, 3], fill_value='extrapolate')


        def rmse(x, y):
            yhat = f(x)
            e = yhat - y
            ssr = np.square(e).sum()
            return ssr/x.size


        rmsea = rmse(xy[:, 0] + dx_loc, dx_loc)
        rmseb = rmse(xy[:, 0] + dx_loc_w, dx_loc_w)
        rmsec = rmse(xy[:, 0] + dx_max_z, dx_max_z)
        plt.plot(xy[:, 0] + dx_loc, dx_loc, '-', label=labels[0] + f', MSE: {rmsea:.2f}')
        plt.plot(xy[:, 0] + dx_loc_w, dx_loc_w, '--', label=labels[1] + f', MSE: {rmseb:.2f}')
        plt.plot(xy[:, 0] + dx_max_z, dx_max_z, '--', label=labels[2] + f', MSE: {rmsec:.2f}')
        plt.xlabel(r'$x$', fontsize=18)
        plt.ylabel(r'$\Delta x$', fontsize=18)
        plt.yticks(np.linspace(-5, 5, num=10))
        plt.xticks(np.linspace(-100, 100, num=9))

        # plt.axis('equal')
        plt.legend(fontsize=18, loc=2)



        # plt.figure(figsize=(10, 6))
        # plt.title('Test ' + str(count+1) + " P= " + str(p))
        plt.subplot(212)
        fig.tight_layout(pad=0.25)

        # find y_min and y_max index
        y_min_ind = np.argmin(data_x_0[:, 1])
        y_max_ind = np.argmax(data_x_0[:, 1])

        xy = np.zeros((n_pred, 2))
        xy[:, 0] = np.linspace(data_x_0[y_min_ind, 0], data_x_0[y_max_ind, 0], num=n_pred)
        xy[:, 1] = np.linspace(data_x_0[y_min_ind, 1], data_x_0[y_max_ind, 1], num=n_pred)

        # calc values
        dx_loc, dy_loc, dz_loc = int_loc.calc_disp(xy, p)
        dx_loc_w, dy_loc_w, dz_loc_w = int_loc_w.calc_disp(xy, p)
        dx_max_loc, dy_max_loc, dz_max_loc = int_max_loc.calc_disp(xy, p)
        dx_max_loc_w, dy_max_loc_w, dz_max_loc_w = int_max_w_loc.calc_disp(xy, p)
        dx_max_z, dy_max_z, dz_max_z = int_max_z.calc_disp(xy, p)
        plt.plot(data_x_0[:, 1] + data_x_0[:, 4], data_x_0[:, 4], '.', label='test')
        f = interp1d(data_x_0[:, 1] + data_x_0[:, 4], data_x_0[:, 4], fill_value='extrapolate')

        rmsea = rmse(xy[:, 1] + dy_loc, dy_loc)
        rmseb = rmse(xy[:, 1] + dy_loc_w, dy_loc_w)
        rmsec = rmse(xy[:, 1] + dy_max_z, dy_max_z)
        plt.plot(xy[:, 1] + dy_loc, dy_loc, '-', label=labels[0] + f', MSE: {rmsea:.2f}')
        plt.plot(xy[:, 1] + dy_loc_w, dy_loc_w, '--', label=labels[1] + f', MSE: {rmseb:.2f}')
        plt.plot(xy[:, 1] + dy_max_z, dy_max_z, '--', label=labels[2] + f', MSE: {rmsec:.2f}')
        plt.xlabel(r'$y$', fontsize=18)
        plt.ylabel(r'$\Delta y$', fontsize=18)
        plt.yticks(np.linspace(-5, 5, num=10))
        plt.xticks(np.linspace(-100, 100, num=9))
        plt.legend(fontsize=18, loc=2)
        # plt.savefig('figs_dx_dy/Test_' + str(count+1) + "_j=" + str(j).zfill(2) + '.png', bbox_inches='tight')
        plt.savefig('figs_dx_dy/Test_' + str(count+1) + "_j=" + str(j).zfill(2) + '.pdf', bbox_inches='tight')
        # break
        plt.close()
    count += 1
        # break
# plt.show()
