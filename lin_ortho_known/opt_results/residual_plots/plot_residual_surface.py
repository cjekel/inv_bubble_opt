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
   

    my_shape = resids.shape
    for j in range(my_shape[1]):
        print(j)
        if j == my_shape[1] - 1:
            x = data[j][0][:, 0]
            y = data[j][0][:, 1]
            z = data[j][0][:, 2]
            dx = resids[0][j]
            dy = resids[1][j]
            dz = resids[2][j]

            fig_dx, ax_dx = plt.subplots(1, 1)
            fig_dy, ax_dy = plt.subplots(1, 1)
            fig_dz, ax_dz = plt.subplots(1, 1)

            cdx = ax_dx.scatter(x, y, s=100, c=dx, marker='o', cmap=plt.cm.plasma)
            fig_dx.colorbar(cdx, ax=ax_dx)
            ax_dx.set_title('Test ' + str(i+1) + r': $\Delta_x$ residual displacement, mm')
            ax_dx.set_xlabel('$x$, mm')
            ax_dx.set_ylabel('$y$, mm')

            cdy = ax_dy.scatter(x, y, s=100, c=dy, marker='o', cmap=plt.cm.plasma)
            fig_dy.colorbar(cdy, ax=ax_dy)
            ax_dy.set_title('Test ' + str(i+1) + r': $\Delta_y$ residual displacement, mm')
            ax_dy.set_xlabel('$x$, mm')
            ax_dy.set_ylabel('$y$, mm')

            cdz = ax_dz.scatter(x, y, s=100, c=dz, marker='o', cmap=plt.cm.plasma)
            fig_dz.colorbar(cdz, ax=ax_dz)
            ax_dz.set_title('Test ' + str(i+1) + r': $\Delta_z$ residual displacement, mm')
            ax_dz.set_xlabel('$x$, mm')
            ax_dz.set_ylabel('$y$, mm')
            # save each fig
            fig_dx.savefig('figs/dx' + str(i) + '.png', bbox_inches='tight', dpi=300)
            fig_dy.savefig('figs/dy' + str(i) + '.png', bbox_inches='tight', dpi=300)
            fig_dz.savefig('figs/dz' + str(i) + '.png', bbox_inches='tight', dpi=300)
        # fig_dx.savefig('dx' + str(i) + '.pdf', bbox_inches='tight')
        # fig_dy.savefig('dy' + str(i) + '.pdf', bbox_inches='tight')
        # fig_dz.savefig('dz' + str(i) + '.pdf', bbox_inches='tight')

        # break
    plt.show()