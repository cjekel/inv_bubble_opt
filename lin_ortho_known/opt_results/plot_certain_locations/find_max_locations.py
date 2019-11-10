import numpy as np
import matplotlib.pyplot as plt

disp = np.load('disp_values_0.npy')
xy = np.load('xy_model.npy')
max_disp_x = np.max(disp[-1, :, 0])
ind_max_disp_x = np.argmax(disp[-1, :, 0])
max_disp_y = np.max(disp[-1, :, 1])
ind_max_disp_y = np.argmax(disp[-1, :, 1])
max_disp_z = np.max(disp[-1, :, 2])
ind_max_disp_z = np.argmax(disp[-1, :, 2])

plt.figure()
plt.plot(np.arange(200), disp[1:, ind_max_disp_x, 0], label='Disp x')
plt.plot(np.arange(200), disp[1:, ind_max_disp_y, 1], label='Disp y')
plt.plot(np.arange(200), disp[1:, ind_max_disp_z, 2], label='Disp z')
plt.legend()
plt.show()

# In [22]: xy[0, ind_max_disp_x, :]
# Out[22]: array([56.25,  0.  ,  0.  ])

# In [23]: xy[0, ind_max_disp_y, :]
# Out[23]: array([ 0. , 62.5,  0. ])

# In [24]: xy[0, ind_max_disp_z, :]
# Out[24]: array([0., 0., 0.])
