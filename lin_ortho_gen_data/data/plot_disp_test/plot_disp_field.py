import os
import numpy as np
import matplotlib.pyplot as plt
import invbubble
from sklearn.preprocessing import MinMaxScaler
homeuser = os.path.expanduser('~')

data = np.load('disp_values_0.npy')
xy_data = np.load('xy_model.npy')
x = xy_data[0, :, 0]
y = xy_data[0, :, 1]
xy = xy_data[0, :, :2]

# new xy data to generate predictions
n_c = 56
xh = np.linspace(-100, 100, n_c)
yh = np.linspace(-100, 100, n_c)
xh, yh = np.meshgrid(xh, yh)
xyh = np.zeros((n_c*n_c, 2))
xyh[:, 0] = xh.flatten()
xyh[:, 1] = yh.flatten()
# np.nan values with r > 100
# xtemp = np.where(xyh[:, 0]**2 + xyh[:, 1]**2 > 100., np.nan)
inds = np.argwhere(xyh[:, 0]**2 + xyh[:, 1]**2 > 100.**2)

# fit rbf and edvaulte
dxh = invbubble.rbf_function(xy_data[0, :, :2], data[-1, :, 0], xyh)
dyh = invbubble.rbf_function(xy_data[0, :, :2], data[-1, :, 1], xyh)
dzh = invbubble.rbf_function(xy_data[0, :, :2], data[-1, :, 2], xyh)

# set outside bounds to 0.0
dxh[inds] = 0.
dyh[inds] = 0.
dzh[inds] = 0.

scaler = MinMaxScaler()


# reshape to plot contours
image = np.zeros((n_c*n_c, 3))
image[:, 0] = dxh
image[:, 1] = dyh
image[:, 2] = dzh
image = scaler.fit_transform(image)
dxh = dxh.reshape(n_c, n_c)
dyh = dyh.reshape(n_c, n_c)
dzh = dzh.reshape(n_c, n_c)

xh = xh.reshape(n_c, n_c)
yh = yh.reshape(n_c, n_c)

# plot contours
plt.figure()
plt.title('dx')
plt.contourf(xh, yh, dxh)
plt.colorbar()
plt.figure()
plt.title('dy')
plt.contourf(xh, yh, dyh)
plt.colorbar()
plt.figure()
plt.title('dz')
plt.contourf(xh, yh, dzh)
plt.colorbar()

# try to do imshow
# image = np.zeros((n_c, n_c, 3))
# image[:, :, 0] = dxh
# image[:, :, 1] = dyh
# image[:, :, 2] = dzh

image = image.reshape(n_c, n_c, 3)

# image.min()

plt.figure()
plt.title('dx')
plt.imshow(image[:, :, 0])

plt.figure()
plt.title('dy')
plt.imshow(image[:, :, 1])

plt.figure()
plt.title('dz')
plt.imshow(image[:, :, 2])

plt.figure()
plt.imshow(image)

plt.show()
