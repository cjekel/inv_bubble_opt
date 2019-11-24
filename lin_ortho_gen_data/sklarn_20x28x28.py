import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

np.random.seed(1231231)
# load the XY data
with h5py.File('10k20x28x28data.hdf5', 'r') as f:
    a_group_key = list(f.keys())[0]
    X = f[a_group_key][()]
scaler = MinMaxScaler()
Y = np.load('mydoe.npy')
num_predictions = Y.shape[1]

print('Original shape:', X.shape, Y.shape)
# reshape X for scikit-learn
X = X.reshape((10000, 20*28*28*3))
# split the data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.99,
                                                    random_state=42)


print('Train shape:', X_train.shape, y_train.shape)

print('x_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')


def compute_scores(modelx, modely, modelz, test_data):
    # compute mae in each component
    y_hatx = modelx.predict(test_data)
    y_haty = modely.predict(test_data)
    y_hatz = modelz.predict(test_data)
    y_hat = np.array((y_hatx, y_haty, y_hatz)).T
    e = np.abs(y_hat-y_test)
    score = e.mean(axis=0)
    print('E1 mae:', score[0])
    print('E2 mae:', score[1])
    if num_predictions == 3:
        print('G12 mae:', score[2])

    # compute mean absolute percentage error
    e_m = np.abs((y_hat - y_test)/y_test)
    e_mape = e_m.mean(axis=0)
    print('E1 mape:', e_mape[0]*100)
    print('E2 mape:', e_mape[1]*100)
    if num_predictions == 3:
        print('G12 mape:', e_mape[2]*100)

    emax = np.max(e, axis=0)
    print('E1 max absolute error:', emax[0])
    print('E2 max absolute error:', emax[1])
    if num_predictions == 3:
        print('G12 max absolute error:', score[2])


# polynomial transform
poly = PolynomialFeatures(degree=1)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# polynomial model
clfx = linear_model.LinearRegression(fit_intercept=False, n_jobs=16)
clfy = linear_model.LinearRegression(fit_intercept=False, n_jobs=16)
clfz = linear_model.LinearRegression(fit_intercept=False, n_jobs=16)

clfx.fit(X_train_poly, y_train[:, 0])
clfy.fit(X_train_poly, y_train[:, 1])
clfz.fit(X_train_poly, y_train[:, 2])

compute_scores(clfx, clfy, clfz, X_test_poly)

# # polynomial transform
# poly2 = PolynomialFeatures(degree=2)
# X_train_poly2 = poly2.fit_transform(X_train)

# # polynomial model
# clfx2 = linear_model.LinearRegression(fit_intercept=False, n_jobs=16)
# clfy2 = linear_model.LinearRegression(fit_intercept=False, n_jobs=16)
# clfz2 = linear_model.LinearRegression(fit_intercept=False, n_jobs=16)

# clfx2.fit(X_train_poly2, y_train[:, 0])
# clfy2.fit(X_train_poly2, y_train[:, 1])
# clfz2.fit(X_train_poly2, y_train[:, 2])

# # need to do the evaluation in batches :o
# # X_test_poly2 = poly.transform(X_test)
# batch_size = 1000
# start = 0
# end = 1000
# for i in range(10):
#     compute_scores(clfx2, clfy2, clfz2, poly2.transform(X_test[start:end]))
#     start += batch_size
#     end += batch_size

# # support vector regression
# svrx = SVR(gamma='scale')
# svry = SVR(gamma='scale')
# svrz = SVR(gamma='scale')

# svrx.fit(X_train, y_train[:, 0])
# svry.fit(X_train, y_train[:, 1])
# svrz.fit(X_train, y_train[:, 2])

# compute_scores(svrx, svry, svrz, X_test)

# gaussian process
kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
gprx = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
gpry = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
gprz = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)

gprx.fit(X_train, y_train[:, 0])
gpry.fit(X_train, y_train[:, 1])
gprz.fit(X_train, y_train[:, 2])

compute_scores(gprx, gpry, gprz, X_test)

# model = Sequential()
# model.add(Conv2D(32, kernel_size=(3, 3),
#                  activation='relu',
#                  input_shape=input_shape))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.025))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# # model.add(Dense(256, activation='relu'))
# # model.add(Dense(128, activation='relu'))

# model.add(Dropout(0.025))
# model.add(Dense(num_predictions, activation='linear'))

# model.compile(loss=keras.losses.mean_absolute_error,
#               optimizer=keras.optimizers.Adam(learning_rate=1e-4),
#               metrics=['mse'])

# # model.compile(loss=keras.losses.categorical_crossentropy,
# #               optimizer=keras.optimizers.Adadelta(),
# #               metrics=['accuracy'])

# history = model.fit(X_train, y_train,
#                     epochs=epochs,
#                     batch_size=batch_size,
#                     validation_data=(X_test, y_test))
# # compute mae in each component
# y_hat = model.predict(X_test)
# e = np.abs(y_hat-y_test)
# score = e.mean(axis=0)
# print('E1 mae:', score[0])
# print('E2 mae:', score[1])
# if num_predictions == 3:
#     print('G12 mae:', score[2])

# # compute mean absolute percentage error
# e_m = np.abs((y_hat - y_test)/y_test)
# e_mape = e_m.mean(axis=0)
# print('E1 mape:', score[0]*100)
# print('E2 mape:', score[1]*100)
# if num_predictions == 3:
#     print('G12 mape:', score[2]*100)

# emax = np.max(e, axis=0)
# print('E1 max absolute error:', emax[0])
# print('E2 max absolute error:', emax[1])
# if num_predictions == 3:
#     print('G12 max absolute error:', score[2])

