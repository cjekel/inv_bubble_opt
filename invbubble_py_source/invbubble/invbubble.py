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
import os
import smtplib
import numpy as np
import pandas as pd
from scipy.interpolate import Rbf
from scipy import linalg
from scipy.spatial.distance import cdist, pdist, squareform
from joblib import Parallel, delayed


def my_exception(ex):
    print('!!!!! *** Exception *** !!!!!')
    print(ex)


class Interpolate(object):

    def __init__(self, X, Disp):
        np, n_nodes, dim = X.shape
        # X is of the form [pressures, nodes, [x y p]]
        # I need to construct P * 3 rbf models for each disp
        self.p = X[:, 0, 2]  # pressures
        self.rbf_models_dx = []
        self.rbf_models_dy = []
        self.rbf_models_dz = []
        for i in range(np):
            x = X[i, :, 0]
            y = X[i, :, 1]
            self.rbf_models_dx.append(Rbf(x, y, Disp[i, :, 0],
                                          function='linear'))
            self.rbf_models_dy.append(Rbf(x, y, Disp[i, :, 1],
                                          function='linear'))
            self.rbf_models_dz.append(Rbf(x, y, Disp[i, :, 2],
                                          function='linear'))

    def calc_disp(self, Xnew, p):
        # check for equality
        ind = np.argwhere(p == self.p)
        if ind.size == 0:
            # check for greater than
            ub = np.argmax(self.p > p)
            lb = ub - 1
            if lb == -1:
                print('Error: Extrapolation not allowed!!!')
                raise np.linalg.LinAlgError
            dp = self.p[ub] - self.p[lb]
            pt = p - self.p[lb]
            # linear interpolation formula
            # (x - x0) / (p - p0) =  (x1 - x0) / (p1 - p0)
            x1 = self.rbf_models_dx[ub](Xnew[:, 0], Xnew[:, 1])
            x0 = self.rbf_models_dx[lb](Xnew[:, 0], Xnew[:, 1])
            dx = ((pt*(x1 - x0)) / dp) + x0
            x1 = self.rbf_models_dy[ub](Xnew[:, 0], Xnew[:, 1])
            x0 = self.rbf_models_dy[lb](Xnew[:, 0], Xnew[:, 1])
            dy = ((pt*(x1 - x0)) / dp) + x0
            x1 = self.rbf_models_dz[ub](Xnew[:, 0], Xnew[:, 1])
            x0 = self.rbf_models_dz[lb](Xnew[:, 0], Xnew[:, 1])
            dz = ((pt*(x1 - x0)) / dp) + x0

        else:
            dx = self.rbf_models_dx[ind[0, 0]](Xnew[:, 0], Xnew[:, 1])
            dy = self.rbf_models_dy[ind[0, 0]](Xnew[:, 0], Xnew[:, 1])
            dz = self.rbf_models_dz[ind[0, 0]](Xnew[:, 0], Xnew[:, 1])
        return dx, dy, dz

    def calc_delta(self, X_new, Ps, Disp_new):
        # for numerical model
        # calculate the average deviation for each p in Ps
        dx_delta = np.zeros(len(Ps))
        dy_delta = np.zeros(len(Ps))
        dz_delta = np.zeros(len(Ps))
        for i, p in enumerate(Ps):
            dx_new, dy_new, dz_new = self.calc_disp(X_new[i], p)
            dx_delta[i] = np.nanmean(np.abs(dx_new - Disp_new[i, :, 0]))
            dy_delta[i] = np.nanmean(np.abs(dy_new - Disp_new[i, :, 1]))
            dz_delta[i] = np.nanmean(np.abs(dz_new - Disp_new[i, :, 2]))
        return dx_delta, dy_delta, dz_delta

    def calc_delta_test(self, X_new, Ps):
        # for test data from bubble test
        # calculate the average deviation for each p in Ps
        dx_delta = np.zeros(len(Ps))
        dy_delta = np.zeros(len(Ps))
        dz_delta = np.zeros(len(Ps))
        for i, p in enumerate(Ps):
            newX = X_new[i][:, :2]
            Disp_new = X_new[i][:, 3:]
            dx_new, dy_new, dz_new = self.calc_disp(newX, p)
            dx_delta[i] = np.nanmean(np.abs(dx_new - Disp_new[:, 0]))
            dy_delta[i] = np.nanmean(np.abs(dy_new - Disp_new[:, 1]))
            dz_delta[i] = np.nanmean(np.abs(dz_new - Disp_new[:, 2]))
        return dx_delta, dy_delta, dz_delta


class InterpolateThenRBF(object):

    def __init__(self, X, Disp):
        # np, n_nodes, dim = X.shape
        # X is of the form [pressures, nodes, [x y p]]
        # I need to construct P * 3 rbf models for each disp
        self.p = X[:, 0, 2]  # pressures
        self.x = X[0, :, 0]
        self.y = X[0, :, 1]
        self.Disp = Disp
        self.rbf_models_dx = []
        self.rbf_models_dy = []
        self.rbf_models_dz = []

    def add_rbfs(self, i):
        # i is the index of the FEM model
        self.rbf_models_dx.append(Rbf(self.x, self.y, self.Disp[i, :, 0],
                                      function='linear'))
        self.rbf_models_dy.append(Rbf(self.x, self.y, self.Disp[i, :, 1],
                                      function='linear'))
        self.rbf_models_dz.append(Rbf(self.x, self.y, self.Disp[i, :, 2],
                                      function='linear'))

    def add_rbfs_delta(self, delta):
        # i is the index of the FEM model
        self.rbf_models_dx.append(Rbf(self.x, self.y, delta[:, 0],
                                      function='linear'))
        self.rbf_models_dy.append(Rbf(self.x, self.y, delta[:, 1],
                                      function='linear'))
        self.rbf_models_dz.append(Rbf(self.x, self.y, delta[:, 2],
                                      function='linear'))

    def calc_disp(self, Xnew, p):
        # check for equality
        ind = np.argwhere(p == self.p)
        if ind.size == 0:
            # check for greater than
            ub = np.argmax(self.p > p)
            lb = ub - 1
            if lb == -1:
                print('Error: Extrapolation not allowed!!!')
                raise np.linalg.LinAlgError
            dp = self.p[ub] - self.p[lb]
            pt = p - self.p[lb]
            # linear interpolation formula
            # (x - x0) / (p - p0) =  (x1 - x0) / (p1 - p0)

            # linearlly interoplate the FE model
            x1 = self.Disp[ub, :, :]
            x0 = self.Disp[lb, :, :]
            delta = ((pt*(x1 - x0)) / dp) + x0
            # fit RBFs
            self.add_rbfs_delta(delta)
        else:
            # first fit rbf to model, then evaulate rbf
            self.add_rbfs(ind)
        # calc dx dy dz
        dx = self.rbf_models_dx[-1](Xnew[:, 0], Xnew[:, 1])
        dy = self.rbf_models_dy[-1](Xnew[:, 0], Xnew[:, 1])
        dz = self.rbf_models_dz[-1](Xnew[:, 0], Xnew[:, 1])
        return dx, dy, dz

    def calc_delta(self, X_new, Ps, Disp_new):
        # for numerical model
        # calculate the average deviation for each p in Ps
        dx_delta = np.zeros(len(Ps))
        dy_delta = np.zeros(len(Ps))
        dz_delta = np.zeros(len(Ps))
        for i, p in enumerate(Ps):
            dx_new, dy_new, dz_new = self.calc_disp(X_new[i], p)
            dx_delta[i] = np.nanmean(np.abs(dx_new - Disp_new[i, :, 0]))
            dy_delta[i] = np.nanmean(np.abs(dy_new - Disp_new[i, :, 1]))
            dz_delta[i] = np.nanmean(np.abs(dz_new - Disp_new[i, :, 2]))
        return dx_delta, dy_delta, dz_delta

    def calc_delta_test(self, X_new, Ps):
        # for test data from bubble test
        # calculate the average deviation for each p in Ps
        dx_delta = np.zeros(len(Ps))
        dy_delta = np.zeros(len(Ps))
        dz_delta = np.zeros(len(Ps))
        for i, p in enumerate(Ps):
            newX = X_new[i][:, :2]
            Disp_new = X_new[i][:, 3:]
            dx_new, dy_new, dz_new = self.calc_disp(newX, p)
            dx_delta[i] = np.nanmean(np.abs(dx_new - Disp_new[:, 0]))
            dy_delta[i] = np.nanmean(np.abs(dy_new - Disp_new[:, 1]))
            dz_delta[i] = np.nanmean(np.abs(dz_new - Disp_new[:, 2]))
        return dx_delta, dy_delta, dz_delta


def rbf_function(xi, di, xih):
    "simple rbf linear function inspired by scipy.interploate.rbf"
    A = squareform(pdist(xi, 'euclidean'))  # Pairwise norm
    nodes = linalg.solve(A, di)
    r = cdist(xih, xi, 'euclidean')
    res = np.dot(r, nodes)
    return res


class InterpolateSimpleRBF(object):

    def __init__(self, X, Disp):
        # np, n_nodes, dim = X.shape
        # X is of the form [pressures, nodes, [x y p]]
        # I need to construct P * 3 rbf models for each disp
        self.p = X[:, 0, 2]  # pressures
        self.x = X[0, :, 0]
        self.y = X[0, :, 1]
        self.xy = X[0, :, :2]
        self.Disp = Disp

    def calc_disp(self, Xnew, p):
        # check for equality
        ind = np.argwhere(p == self.p)
        if ind.size == 0:
            # check for greater than
            ub = np.argmax(self.p > p)
            lb = ub - 1
            if lb == -1:
                print('Error: Extrapolation not allowed!!!')
                raise np.linalg.LinAlgError
            dp = self.p[ub] - self.p[lb]
            pt = p - self.p[lb]
            # linear interpolation formula
            # (x - x0) / (p - p0) =  (x1 - x0) / (p1 - p0)

            # linearlly interoplate the FE model
            x1 = self.Disp[ub, :, :]
            x0 = self.Disp[lb, :, :]
            delta = ((pt*(x1 - x0)) / dp) + x0
            dx = rbf_function(self.xy, delta[:, 0], Xnew[:, :2])
            dy = rbf_function(self.xy, delta[:, 1], Xnew[:, :2])
            dz = rbf_function(self.xy, delta[:, 2], Xnew[:, :2])
        else:
            dx = rbf_function(self.xy, ind[:, 0], Xnew[:, :2])
            dy = rbf_function(self.xy, ind[:, 1], Xnew[:, :2])
            dz = rbf_function(self.xy, ind[:, 2], Xnew[:, :2])
        return dx, dy, dz

    def calc_delta_test_par(self, i):
        p = self.Ps[i]
        newX = self.X_new[i][:, :2]
        Disp_new = self.X_new[i][:, 3:]
        dx_new, dy_new, dz_new = self.calc_disp(newX, p)
        dx_delta = np.nanmean(np.abs(dx_new - Disp_new[:, 0]))
        dy_delta = np.nanmean(np.abs(dy_new - Disp_new[:, 1]))
        dz_delta = np.nanmean(np.abs(dz_new - Disp_new[:, 2]))
        return dx_delta, dy_delta, dz_delta

    def calc_delta_test(self, X_new, Ps):
        # for test data from bubble test
        # calculate the average deviation for each p in Ps
        pn = len(Ps)
        self.X_new = X_new
        self.Ps = Ps
        dx = Parallel(n_jobs=-1, backend='loky')(delayed(self.calc_delta_test_par)(i) for i in range(pn))  # noqa E501
        dx = np.array(dx)
        return dx[:, 0], dx[:, 1], dx[:, 2]


def write_material_model(x):
    with open('model_template.inp', 'r') as d:
        with open('model.inp', 'w') as f:
            data = d.read()
            data = data.replace('E1_orth', str(x[0]))
            data = data.replace('E2_orth', str(x[1]))
            data = data.replace('G12_orth', str(x[2]/100.))
            data = data.replace('E23_orth', str(x[1]/2.48))
            f.write(data)


def write_iso_two_model(x):
    with open('model_template.inp', 'r') as d:
        with open('model.inp', 'w') as f:
            data = d.read()
            data = data.replace('E1_iso', str(x[0]))
            nu = (x[0]/(2.0*x[1]*1e-1)) - 1.0
            data = data.replace('Nu12_iso', str(nu))
            f.write(data)


def write_iso_one_model(x):
    with open('model_template.inp', 'r') as d:
        with open('model.inp', 'w') as f:
            data = d.read()
            data = data.replace('E1_iso', str(x[0]))
            f.write(data)


def run_model():
    abqcommand = 'abq job=model interactive cpus=4 ask_delete=OFF > /dev/null'
    if os.name == 'nt':
        abqcommand = 'abq job=model interactive cpus=4 ask_delete=OFF > NUL'
    val = os.system(abqcommand)
    # on linux val == 0 when success
    return val


def read_sta():
    try:
        with open('model.sta', 'r') as f:
            read_data = f.readlines()
            print(read_data[-1])
        if read_data[-1] == ' THE ANALYSIS HAS COMPLETED SUCCESSFULLY\n':
            # the analysis was completed successfully
            success = True
        else:
            success = False
    except Exception as ex:
        my_exception(ex)
        success = False
    return success


def export_csv_files():
    abq_command = 'abaqus cae noGUI=export_csv_files.py > /dev/null'
    if os.name == 'nt':
        abq_command = 'abaqus cae noGUI=export_csv_files.py > NUL'
    val = os.system(abq_command)
    return val


def read_csv_files(save=False):
    # initiate array of zeros
    node_values = np.zeros((201, 937, 3))

    for i in range(200):
        file_name = 'BubbleTest/L00' + str(i).zfill(3) + '.csv'
        temp = pd.read_csv(file_name, delimiter=',')
        # grab the x values
        node_values[i + 1, :, 0] = temp.values[:, 11]
        # grab the y values
        node_values[i + 1, :, 1] = temp.values[:, 12]
        # grab the z values
        node_values[i + 1, :, 2] = temp.values[:, 13]

    node_values[0, :, 0] = node_values[1, :, 0]
    node_values[0, :, 1] = node_values[1, :, 1]
    node_values[0, :, 2] = node_values[1, :, 2]

    # load the x, y initial values
    file_name = 'BubbleTest/L00' + str(0).zfill(3) + '.csv'
    temp = pd.read_csv(file_name, delimiter=',')
    load = np.zeros(201)
    tempload = np.linspace(0.001, 3.0, 200)
    load[1:] = load[1:] + tempload
    load = load * 0.0001

    X = np.zeros((201, 937, 3))

    X[:, :, 0] = temp.values[:, 5]
    X[:, :, 1] = temp.values[:, 6]
    for i in range(200):
        X[i+1, :, 2] = load[i+1]

    if save is True:
        # save files
        np.save('xy_model.npy', X)
        np.save('disp_values.npy', node_values)
    return X, node_values


def delete_files():
    files_to_remove = ['model.com', 'model.dat', 'model.msg', 'model.odb',
                       'model.prt', 'model.sim', 'model.sta', 'model.lck',
                       'model.simdir', 'model.*.SMABulk', '*.SMASim',
                       '*SMABulk*', '*SMAS*']
    for f in files_to_remove:
        try:
            os.remove(f)
        except FileNotFoundError:
            pass
        except Exception as ex:  # noqa E841
            pass  # I don't need to see errors when files aren't deleted
            # my_exception(ex)


class BubbleOpt(object):

    def __init__(self, opt_hist_file, header, max_obj, xdata_fn, ydata_fn,
                 test_data=[], mat_model='lin-ortho', debug=False,
                 MyInt=InterpolateSimpleRBF, weights=[1.0, 1.0, 1.0],
                 residual_fn=False):
        self.weights = np.array(weights)
        # header should be something like E1, E2, G12, OBJ, Fail
        self.opt_hist_file = opt_hist_file
        self.max_obj = max_obj
        self.run = 0
        self.mydf = pd.DataFrame(data=None, index=None, columns=header,
                                 dtype=np.float)
        self.header = header
        # if you are running on real test data, then you don't need
        # either xdata or ydata
        if xdata_fn is not None:
            self.Xdata = np.load(xdata_fn)
        if ydata_fn is not None:
            self.Ydata = np.load(ydata_fn)
        self.test_data = test_data
        self.n_test_data = len(test_data)
        # material model choices
        # mat_model = ['lin-ortho', 'iso-two', 'iso-one']
        self.mat_model = mat_model
        self.debug = debug
        self.MyInt = MyInt
        self.residual_fn = residual_fn

    def update_df(self, x, my_obj, suc):
        # update and save dataframe
        if len(x) == 3:
            self.mydf = self.mydf.append([{self.header[0]: x[0],
                                           self.header[1]: x[1],
                                           self.header[2]: x[2],
                                           self.header[3]: my_obj,
                                           self.header[4]: float(suc)}])
        elif len(x) == 2:
            self.mydf = self.mydf.append([{self.header[0]: x[0],
                                           self.header[1]: x[1],
                                           self.header[2]: my_obj,
                                           self.header[3]: float(suc)}])
        elif len(x) == 1:
            self.mydf = self.mydf.append([{self.header[0]: x[0],
                                           self.header[1]: my_obj,
                                           self.header[2]: float(suc)}])
        self.max_obj = self.mydf.values[:, -2].max()
        self.mydf.to_csv(self.opt_hist_file)
        self.run += 1

    def run_abq_model(self, x):
        val = 1
        # write the material constants
        if self.mat_model == 'lin-ortho':
            write_material_model(x)
            # run the finite element model
            val = run_model()
        elif self.mat_model == 'iso-two':
            # run the finite element model
            nu = (x[0]/(2.0*x[1]*1e-1)) - 1.0
            # only run the model if poisson's ratio is valid
            if nu <= 0.5 and nu >= 0.0:
                write_iso_two_model(x)
                val = run_model()
        elif self.mat_model == 'iso-one':
            write_iso_one_model(x)
            # run the finite element model
            val = run_model()
        else:
            print('You have assigned an improper material model!')
            raise ValueError
        if val == 0:
            # check the status file to ensure the FE model was successful
            suc = read_sta()
        else:
            suc = False
        return val, suc

    def debug_obj(self, dx_delta, dy_delta, dz_delta):
        print('dx delta', np.nansum(dx_delta))
        print('dy delta', np.nansum(dy_delta))
        print('dz delta', np.nansum(dz_delta))

    def calc_obj_function_abq_data(self, x, run_abq=True):
        # run_abq option to turn on or off abaqus model running
        try:
            if run_abq is True:
                val, suc = self.run_abq_model(x)
            else:
                suc = True
                val = 0
            if suc is True:
                if run_abq is True:
                    # export the csv files of node displacements
                    val = export_csv_files()
                if val == 0:
                    suc = True
                    # read the csv files of node displacements
                    X, Disp = read_csv_files(save=False)
                    # fit interpolation model
                    my_int = self.MyInt(X, Disp)
                    dx_delta, dy_delta, dz_delta = my_int.calc_delta(self.Xdata[:, :, :2],  # noqa E501
                                                                     self.Xdata[:, 0, 2],  # noqa E501
                                                                     self.Ydata)  # noqa E501
                    delete_files()
                    mydx = np.array((dx_delta, dy_delta, dz_delta))
                    my_obj = np.nansum(self.weights*mydx)
                    self.update_df(x, my_obj, suc)
                    if self.debug is True:
                        self.debug_obj(dx_delta, dy_delta, dz_delta)
                    return my_obj
            else:
                delete_files()
                self.update_df(x, self.max_obj, suc)
                return self.max_obj
        except Exception as ex:
            my_exception(ex)
            delete_files()
            self.update_df(x, self.max_obj, False)
            return self.max_obj

    def calc_obj_function_test_data(self, x, run_abq=True):
        # run_abq option to turn on or off abaqus model running
        try:
            if run_abq is True:
                val, suc = self.run_abq_model(x)
            else:
                suc = True
                val = 0
            if suc is True:
                if run_abq is True:
                    # export the csv files of node displacements
                    val = export_csv_files()
                if val == 0:
                    suc = True
                    # read the csv files of node displacements
                    X, Disp = read_csv_files(save=False)
                    # fit interpolation model
                    my_int = self.MyInt(X, Disp)
                    dx = np.zeros(self.n_test_data)
                    dy = dx.copy()
                    dz = dx.copy()
                    for i in range(self.n_test_data):
                        dx_delta, dy_delta, dz_delta = my_int.calc_delta_test(self.test_data[i][:, 0],  # noqa E501
                                                                              self.test_data[i][:, 1])  # noqa E501
                        if self.residual_fn is not False:
                            np.save(self.residual_fn, [dx_delta, dy_delta,
                                                       dz_delta])
                        dx[i] = np.nanmean(dx_delta)
                        dy[i] = np.nanmean(dy_delta)
                        dz[i] = np.nanmean(dz_delta)
                    delete_files()
                    mydx = np.array((dx.mean(), dy.mean(), dz.mean()))
                    my_obj = np.nansum(self.weights*mydx)
                    self.update_df(x, my_obj, suc)
                    if self.debug is True:
                        self.debug_obj(dx.mean(), dy.mean(), dz.mean())
                        print('Individual results')
                        print('dx', dx)
                        print('dy', dy)
                        print('dz', dz)
                        print('weight*dz', self.weights[2]*dz)
                    return my_obj
            else:
                delete_files()
                self.update_df(x, self.max_obj, suc)
                return self.max_obj
        except Exception as ex:
            my_exception(ex)
            delete_files()
            self.update_df(x, self.max_obj, False)
            return self.max_obj


def send_email(receiver_email, subject, message):
    # get user and email details
    homeuser = os.path.expanduser('~')
    with open(os.path.join(homeuser, 'yandex.info')) as f:
        lines = f.readlines()
        sender_email = lines[0].strip('\n')
        password = lines[1].strip('\n')

    # print(sender_email, password)

    port = 465  # For starttls
    smtp_server = "smtp.yandex.com"
    sender_email = "work.complete@yandex.com"
    my_message = 'Subject: ' + subject + '\n\n' + message

    with smtplib.SMTP_SSL(smtp_server, port) as server:
        server.ehlo()  # Can be omitted
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, my_message)
