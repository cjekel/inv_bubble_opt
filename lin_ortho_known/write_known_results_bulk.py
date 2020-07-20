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
from invbubble import *


material_models = [[0.34281612, 0.24753703, 0.47521649, 'max_z_01'],
                   [0.21305367, 0.21881558, 0.26647432, 'max_z_02'],
                   [0.20647327, 0.20898952, 0.26260512, 'max_z_03'],
                   [0.20966416, 0.21375401, 0.26652669, 'max_z_04']]
                #    [0.17175943, 0.15505512, 0.28500203, 'max_loc_w_01'],
                #    [0.20066197, 0.20031188, 0.27306079, 'max_loc_w_02'],
                #    [0.16975526, 0.16633156, 0.33922213, 'max_loc_w_03'],
                #    [0.15971755, 0.16016168, 0.36973612, 'max_loc_w_04'],
                #    [0.22769496, 0.16392157, 0.60330603, 'max_loc_01'],
                #    [0.14283635, 0.10568726, 0.44515912, 'max_loc_02'],
                #    [0.22663274, 0.18564068, 0.46086686, 'max_loc_03'],
                #    [0.16154278, 0.13000562, 0.37468001, 'max_loc_04'],
                #    [0.22375599, 0.23479666, 0.27276717, 'loc_01'],
                #    [0.29727554, 0.23081033, 0.44889951, 'loc_02'],
                #    [0.28382458, 0.22378443, 0.4474975, 'loc_03'],
                #    [0.28004711, 0.24353676, 0.32121493, 'loc_04'],
                #    [0.30297657, 0.2286971,  0.47535454, 'loc_w_01'],
                #    [0.30399262, 0.22917759, 0.47698241, 'loc_w_02'],
                #    [0.30419471, 0.22927316, 0.47632377, 'loc_w_03'],
                #    [0.30461725, 0.22947298, 0.47608423, 'loc_w_04']]


for test in material_models:
    filename = test[-1] + '.npy'

    x = [test[0], test[1], test[2]]
    write_material_model(x)

    run_model()
    suc = read_sta()
    if suc is True:
        export_csv_files()
        X, Disp = read_csv_files(save=True, fn=filename)
    delete_files()
