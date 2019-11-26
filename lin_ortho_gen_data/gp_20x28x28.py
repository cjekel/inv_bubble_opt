import h5py
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
# from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import GPy

np.random.seed(1231231)
# load the XY data
with h5py.File('1k20x28x28data.hdf5', 'r') as f:
    a_group_key = list(f.keys())[0]
    X = f[a_group_key][()]

# optional load DIC test data
scaler = joblib.load('minmaxscaler_20x28x28_1k.z')
X_test_data = np.load('test_data.npy')
print("test data images max min", X_test_data.max(), X_test_data.min())
# transform with the min max scaler from the fea models
X_test_data = X_test_data.reshape((4*20*28*28, 3))
X_test_data = scaler.transform(X_test_data)
X_test_data = X_test_data.reshape((4, 20*28*28*3))
print("test data images max min", X_test_data.max(), X_test_data.min())

Y = np.load('my_success_runs_doe.npy')
# Due to scaling in G12, real value is less
Y[:, 2] *= 1e-2
num_predictions = Y.shape[1]

print('Original shape:', X.shape, Y.shape)
# reshape X for scikit-learn
X = X.reshape((477, 20*28*28*3))
# split the data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1,
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


def compute_gp_scores(model, test_data, y_test):
    # compute mae in each component
    y_hat, _ = model.predict(test_data)
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


# # polynomial transform; not as good as GP
# poly = PolynomialFeatures(degree=1)
# X_train_poly = poly.fit_transform(X_train)
# X_test_poly = poly.transform(X_test)
# # X_test_data_poly = poly.transform(X_test_data)

# # polynomial model
# clfx = linear_model.LinearRegression(fit_intercept=False, n_jobs=16)
# clfy = linear_model.LinearRegression(fit_intercept=False, n_jobs=16)
# clfz = linear_model.LinearRegression(fit_intercept=False, n_jobs=16)

# clfx.fit(X_train_poly, y_train[:, 0])
# clfy.fit(X_train_poly, y_train[:, 1])
# clfz.fit(X_train_poly, y_train[:, 2])

# compute_scores(clfx, clfy, clfz, X_test_poly)

# # predictions on DIC test data
# E1hat = clfx.predict(X_test_data_poly)
# E2hat = clfy.predict(X_test_data_poly)
# G12hat = clfz.predict(X_test_data_poly)

# print('Predictions')
# print('E1:', E1hat)
# print('E2:', E2hat)
# print('G12:', G12hat)

# gaussian process in scikit-learn
# this takes too long, use GPy instead
# kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
# gprx = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
# gpry = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
# gprz = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)

# gpry = GPy.models.GPRegression(X,Y,kernel)
# gprz = GPy.models.GPRegression(X,Y,kernel)

# gprx.fit(X_train, y_train[:, 0])
# gpry.fit(X_train, y_train[:, 1])
# gprz.fit(X_train, y_train[:, 2])

# compute_scores(gprx, gpry, gprz, X_test)

# # predictions on DIC test data
# E1hat_gp = gprx.predict(X_test_data)
# E2hat_gp = gpry.predict(X_test_data)
# G12hat_gp = gprz.predict(X_test_data)

# print('Predictions')
# print('E1:', E1hat_gp)
# print('E2:', E2hat_gp)
# print('G12:', G12hat_gp)

Kernel = GPy.kern.Exponential(input_dim=X_train.shape[1])
gp = GPy.models.GPRegression(X_train, y_train, Kernel)
gp.optimize_restarts(num_restarts=5)
# compute scores on training data
compute_gp_scores(gp, X_train, y_train)

# copute scores on testing data
compute_gp_scores(gp, X_test, y_test)

# predictions on dic test data
material, var = gp.predict(X_test_data)

print('Material\n', material)
print('GP test variance\n', var)
