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
import numpy as np
import pandas as pd
from scipy.optimize import fmin_slsqp
from scipy.interpolate import Rbf


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
            dx_delta[i] = np.abs(dx_new - Disp_new[i, :, 0]).mean()
            dy_delta[i] = np.abs(dy_new - Disp_new[i, :, 1]).mean()
            dz_delta[i] = np.abs(dz_new - Disp_new[i, :, 2]).mean()
        return dx_delta, dy_delta, dz_delta

    def calc_delta_test(self, X_new, Ps):
        # for test data from bubble test
        # calculate the average deviation for each p in Ps
        dx_delta = np.zeros(len(Ps))
        dy_delta = np.zeros(len(Ps))
        dz_delta = np.zeros(len(Ps))
        for i, p in enumerate(Ps):
            print(i, p)
            newX = X_new[i][:, :2]
            Disp_new = X_new[i][:, 3:]
            dx_new, dy_new, dz_new = self.calc_disp(newX, p)
            dx_delta[i] = np.abs(dx_new - Disp_new[:, 0]).mean()
            dy_delta[i] = np.abs(dy_new - Disp_new[:, 1]).mean()
            dz_delta[i] = np.abs(dz_new - Disp_new[:, 2]).mean()
        return dx_delta, dy_delta, dz_delta


def write_material_model(x):
    with open('model_template.inp', 'r') as d:
        with open('model.inp', 'w') as f:
            data = d.read()
            data = data.replace('E1_orth', str(x[0]))
            data = data.replace('E2_orth', str(x[1]))
            data = data.replace('G12_orth', str(x[2]))
            data = data.replace('E23_orth', str(x[1]/2.48))
            f.write(data)


def run_model():
    abqcommand = 'abq2018 job=model interactive cpus=1 ask_delete=OFF'
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
    except:
        success = False
    return success


def export_csv_files():
    abq_command = 'abaqus cae noGUI=export_csv_files.py'
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
                       'model.simdir']
    for f in files_to_remove:
        try:
            os.remove(f)
        except:
            pass


def calc_obj_function(x):
    try:
        # write the material constants
        write_material_model(x)
        # run the finite element model
        val = run_model()
        if val == 0:
            suc = True
        else:
            suc = False
        while suc is True:
            # check the status file to ensure the FE model was successful
            suc = read_sta()
            # export the csv files of node displacements
            val = export_csv_files()
            if val == 0:
                suc = True
            else:
                suc = False
            # read the csv files of node displacements
            X, Disp = read_csv_files(save=False)
            # fit interpolation model
            my_int = Interpolate(X, Disp)
            dx_delta, dy_delta, dz_delta = my_int.calc_delta(X[s1, :, :2], X[s1, 0, 2], Y[s1])
            delete_files()
            return dx_delta.sum() + dy_delta.sum() + dz_delta.sum()
        if suc is False:
            delete_files()
            return max_obj
    except:
        delete_files()
        return max_obj


# Known material model: x = [800.0*1e-3, 150.0*1e-3, 25.0*1e-3]
x = [800.0*1e-3, 150.0*1e-3, 25.0*1e-3]
max_obj = 1000.0
obj = calc_obj_function(x)
