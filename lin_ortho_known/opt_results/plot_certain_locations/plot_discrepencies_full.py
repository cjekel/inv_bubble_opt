import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import Rbf
import invbubble

homeuser = os.path.expanduser('~')

x_max = 56.25
y_max = 62.5
x_zero = 0.0
y_zero = 0.0

# points to evaluate displacements
data_points = np.zeros((3, 2))
data_points[0, 0] = x_max
data_points[0, 1] = y_zero
data_points[1, 0] = x_zero
data_points[1, 1] = y_max
data_points[2, 0] = x_zero
data_points[2, 1] = y_zero

datanames = ['blue00.npy',
             'blue01_rotated_90.npy',
             'blue02_rotated_90.npy',
             'blue03.npy']
xy_model = np.load('xy_model.npy')
models = ['disp_values_0.npy', 'disp_values_1.npy', 'disp_values_2.npy',
          'disp_values_3.npy']
models_w = ['disp_values_0_w.npy', 'disp_values_1_w.npy',
            'disp_values_2_w.npy', 'disp_values_3_w.npy']

for i in range(4):
    data = np.load(os.path.join(homeuser, datanames[i]), allow_pickle=True)
    dx_data = np.load('dx_data_' + datanames[i])
    dy_data = np.load('dy_data_' + datanames[i])
    dz_data = np.load('dz_data_' + datanames[i])
    P = data[:, 1]
    disp = np.load(models[0])
    disp_w = np.load(models_w[0])
    my_inv = invbubble.InterpolateSimpleRBF(xy_model, disp)
    my_inv_w = invbubble.InterpolateSimpleRBF(xy_model, disp_w)
    my_inv_iso_two = invbubble.InterpolateSimpleRBF(xy_model, np.load('../../../iso_two_param/opt_results/displacements/' + models[0]))  # noqa E501
    my_inv_iso_two_w = invbubble.InterpolateSimpleRBF(xy_model, np.load('../../../iso_two_param/opt_results/displacements/' + models_w[0]))  # noqa E501
    my_inv_iso_one = invbubble.InterpolateSimpleRBF(xy_model, np.load('../../../iso_one_param/opt_results/displacements/' + models[0]))  # noqa E501
    my_inv_iso_one_w = invbubble.InterpolateSimpleRBF(xy_model, np.load('../../../iso_one_param/opt_results/displacements/' + models_w[0]))  # noqa E501
    dx = np.zeros(len(P))
    dy = np.zeros_like(dx)
    dz = np.zeros_like(dx)
    dx_w = np.zeros_like(dx)
    dy_w = np.zeros_like(dx)
    dz_w = np.zeros_like(dx)
    dx_one = np.zeros_like(dx)
    dy_one = np.zeros_like(dx)
    dz_one = np.zeros_like(dx)
    dx_one_w = np.zeros_like(dx)
    dy_one_w = np.zeros_like(dx)
    dz_one_w = np.zeros_like(dx)
    dx_two = np.zeros_like(dx)
    dy_two = np.zeros_like(dx)
    dz_two = np.zeros_like(dx)
    dx_two_w = np.zeros_like(dx)
    dy_two_w = np.zeros_like(dx)
    dz_two_w = np.zeros_like(dx)
    for j, p in enumerate(P):
        data_x, data_y, data_z = my_inv.calc_disp(data_points, p)
        dx[j] = data_x[0]
        dy[j] = data_y[1]
        dz[j] = data_z[2]
        data_x, data_y, data_z = my_inv_w.calc_disp(data_points, p)
        dx_w[j] = data_x[0]
        dy_w[j] = data_y[1]
        dz_w[j] = data_z[2]
        data_x, data_y, data_z = my_inv_iso_one.calc_disp(data_points, p)
        dx_one[j] = data_x[0]
        dy_one[j] = data_y[1]
        dz_one[j] = data_z[2]
        data_x, data_y, data_z = my_inv_iso_one_w.calc_disp(data_points, p)
        dx_one_w[j] = data_x[0]
        dy_one_w[j] = data_y[1]
        dz_one_w[j] = data_z[2]
        data_x, data_y, data_z = my_inv_iso_two.calc_disp(data_points, p)
        dx_two[j] = data_x[0]
        dy_two[j] = data_y[1]
        dz_two[j] = data_z[2]
        data_x, data_y, data_z = my_inv_iso_two_w.calc_disp(data_points, p)
        dx_two_w[j] = data_x[0]
        dy_two_w[j] = data_y[1]
        dz_two_w[j] = data_z[2]

    plt.figure()
    # plt.title('Test ' + str(i+1))
    # plt.plot(P*10000., dx_one, 's:', label=r'One Iso. Obj: $e$')
    # plt.plot(P*10000., dx_one_w, 'o:', label=r'One Iso. Obj: $e_w$')
    plt.plot(P*10000., dx_two, 'd--', label=r'Two Iso. Obj: $e$')
    plt.plot(P*10000., dx_two_w, '*--', label=r'Two Iso. Obj: $e_w$')
    plt.plot(P*10000., dx, '+-', label=r'Lin. Ortho. Obj: $e$')
    plt.plot(P*10000., dx_w, '.-', label=r'Lin. Ortho. Obj: $e_w$')
    plt.plot(P*10000., dx_data, 'ok', label='Test data')
    plt.xlabel('Pressure, bar')
    plt.ylabel(r'$\Delta_x$ Displacement, mm')
    plt.legend()
    plt.grid(True)
    plt.savefig('figs/dx_test_' + str(i+1) + '_.png', bbox_inches='tight')
    plt.savefig('figs/dx_test_' + str(i+1) + '_.pdf', bbox_inches='tight')

    plt.figure()
    # plt.title('Test ' + str(i+1))
    # plt.plot(P*10000., dy_one, 's:', label=r'One Iso. Obj: $e$')
    # plt.plot(P*10000., dy_one_w, 'o:', label=r'One Iso. Obj: $e_w$')
    plt.plot(P*10000., dy_two, 'd--', label=r'Two Iso. Obj: $e$')
    plt.plot(P*10000., dy_two_w, '*--', label=r'Two Iso. Obj: $e_w$')
    plt.plot(P*10000., dy, '+-', label=r'Lin. Ortho. Obj: $e$')
    plt.plot(P*10000., dy_w, '.-', label=r'Lin. Ortho. Obj: $e_w$')
    plt.plot(P*10000., dy_data, 'ok', label='Test data')
    plt.xlabel('Pressure, bar')
    plt.ylabel(r'$\Delta_y$ Displacement, mm')
    plt.legend()
    plt.grid(True)
    plt.savefig('figs/dy_test_' + str(i+1) + '_.png', bbox_inches='tight')
    plt.savefig('figs/dy_test_' + str(i+1) + '_.pdf', bbox_inches='tight')

    plt.figure()
    # plt.title('Test ' + str(i+1))
    # plt.plot(P*10000., dz_one, 's:', label=r'One Iso. Obj: $e$')
    # plt.plot(P*10000., dz_one_w, 'o:', label=r'One Iso. Obj: $e_w$')
    plt.plot(P*10000., dz_two, 'd--', label=r'Two Iso. Obj: $e$')
    plt.plot(P*10000., dz_two_w, '*--', label=r'Two Iso. Obj: $e_w$')
    plt.plot(P*10000., dz, '+-', label=r'Lin. Ortho. Obj: $e$')
    plt.plot(P*10000., dz_w, '.-', label=r'Lin. Ortho. Obj: $e_w$')
    plt.plot(P*10000., dz_data, 'ok', label='Test data')
    plt.xlabel('Pressure, bar')
    plt.ylabel(r'$\Delta_z$ Displacement, mm')
    plt.legend()
    plt.grid(True)
    plt.savefig('figs/dz_test_' + str(i+1) + '_.png', bbox_inches='tight')
    plt.savefig('figs/dz_test_' + str(i+1) + '_.pdf', bbox_inches='tight')

plt.show()
