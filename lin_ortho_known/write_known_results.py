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
    abqcommand = 'abq job=model interactive cpus=4 ask_delete=OFF'
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


x = [800.0*1e-3, 150.0*1e-3, 25.0*1e-3]
write_material_model(x)

run_model()
suc = read_sta()
if suc is True:
    export_csv_files()
    X, Disp = read_csv_files(save=True)
delete_files()