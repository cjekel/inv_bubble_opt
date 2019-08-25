import numpy as np
import matplotlib.pyplot as plt
import os

homeuser = os.path.expanduser('~')

filenames = ['residuals_blue00.npy',
             'residuals_blue01.npy',
             'residuals_blue02.npy',
             'residuals_blue03.npy']
datanames = ['blue00.npy',
             'blue01_rotated_90.npy',
             'blue02_rotated_90.npy',
             'blue03.npy']

perfx = np.linspace(-100, 100, 100)
perfy = np.zeros(100)

for i in range(0, 4):
# for residual_file in filenames:
    resids = np.load(filenames[i], allow_pickle=True)
    print(resids.shape)
    data = np.load(os.path.join(homeuser, datanames[i]), allow_pickle=True)
    fig_dx, ax_dx = plt.subplots(2, 1, figsize=(8, 6))
    fig_dy, ax_dy = plt.subplots(2, 1, figsize=(8, 6))
    fig_dz, ax_dz = plt.subplots(2, 1, figsize=(8, 6))
    

    for j, _ in enumerate(resids):
        x = data[j][0][:, 0]
        y = data[j][0][:, 1]
        z = data[j][0][:, 2]
        dx = resids[0][j]
        dy = resids[1][j]
        dz = resids[2][j]

        ax_dx[0].set_title('$\Delta x$ residual' + ' Test number: ' + str(i+1))
        ax_dx[0].plot(x, dx, 'xk', markersize=.5)
        ax_dx[0].plot(perfx, perfy, '-r')
        ax_dx[0].set_xlabel('$x$', fontsize='10')
        ax_dx[0].set_ylabel('$\Delta x$ residual', fontsize='10')
        ax_dx[1].plot(y, dx, 'xk', markersize=.5)
        ax_dx[1].plot(perfx, perfy, '-r')
        ax_dx[1].set_xlabel('$y$', fontsize='10')
        ax_dx[1].set_ylabel('$\Delta x$ residual', fontsize='10')

        ax_dy[0].set_title('$\Delta y$ residual' + ' Test number: ' + str(i+1))
        ax_dy[0].plot(x, dy, 'xk', markersize=.5)
        ax_dy[0].plot(perfx, perfy, '-r')
        ax_dy[0].set_xlabel('$x$', fontsize='10')
        ax_dy[0].set_ylabel('$\Delta y$ residual', fontsize='10')
        ax_dy[1].plot(y, dy, 'xk', markersize=.5)
        ax_dy[1].plot(perfx, perfy, '-r')
        ax_dy[1].set_xlabel('$y$', fontsize='10')
        ax_dy[1].set_ylabel('$\Delta y$ residual', fontsize='10')

        ax_dz[0].set_title('$\Delta z$ residual' + ' Test number: ' + str(i+1))
        ax_dz[0].plot(x, dz, 'xk', markersize=.5)
        ax_dz[0].plot(perfx, perfy, '-r')
        ax_dz[0].set_xlabel('$x$', fontsize='10')
        ax_dz[0].set_ylabel('$\Delta z$ residual', fontsize='10')
        ax_dz[1].plot(y, dz, 'xk', markersize=.5)
        ax_dz[1].plot(perfx, perfy, '-r')
        ax_dz[1].set_xlabel('$y$', fontsize='10')
        ax_dz[1].set_ylabel('$\Delta z$ residual', fontsize='10')
        if i == 1 or i == 2:
            ax_dx[0].set_ylim([-2., 2.])
            ax_dx[1].set_ylim([-2, 2.])
            ax_dy[0].set_ylim([-2., 2.])
            ax_dy[1].set_ylim([-2, 2.])
            ax_dz[0].set_ylim([-10., 10.])
            ax_dz[1].set_ylim([-10, 10.])
        # save each fig
        fig_dx.savefig('dx' + str(i) + '.png', bbox_inches='tight', dpi=300)
        fig_dy.savefig('dy' + str(i) + '.png', bbox_inches='tight', dpi=300)
        fig_dz.savefig('dz' + str(i) + '.png', bbox_inches='tight', dpi=300)
        fig_dx.savefig('dx' + str(i) + '.pdf', bbox_inches='tight')
        fig_dy.savefig('dy' + str(i) + '.pdf', bbox_inches='tight')
        fig_dz.savefig('dz' + str(i) + '.pdf', bbox_inches='tight')

    # break