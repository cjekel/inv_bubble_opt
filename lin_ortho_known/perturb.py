import numpy as np
import pandas as pd
import invbubble
import os


def write_lin_orth_model(mat, preturb=1e-3, random=False):
    a = list(np.arange(0, 9))
    b = list(np.arange(946, 7908))
    skprow = a + b
    my_node_data = pd.read_csv('model_template.inp', skiprows=skprow,
                               names=['node', 'x', 'y', 'z'])
    with open('model_template.inp', 'r') as d:
        with open('model.inp', 'w') as f:
            data = d.readlines()
            for i in range(9, 946):
                if random:
                    xrand = (-preturb - preturb) * np.random.random() + preturb
                    yrand = (-preturb - preturb) * np.random.random() + preturb
                else:
                    xrand = preturb
                    yrand = preturb
                node_number = str(int(my_node_data.values[i-9, 0]))

                x = str(my_node_data.values[i-9, 1] + xrand)
                y = str(my_node_data.values[i-9, 2] + yrand)
                z = str(my_node_data.values[i-9, 3])
                row = [node_number, x, y, z]
                data[i] = ', '.join(row) + ' \n'
            data = ''.join(data)
            data = data.replace('E1_orth', str(mat[0]))
            data = data.replace('E2_orth', str(mat[1]))
            data = data.replace('G12_orth', str(mat[2]/100.))
            data = data.replace('E23_orth', str(mat[1]/2.48))
            f.write(data)


if __name__ == "__main__":
    np.random.seed(121)

    invbubble.delete_files()

    # load the test data
    homeuser = os.path.expanduser('~')
    blue00 = np.load(os.path.join(homeuser, 'blue00.npy'))
    blue01 = np.load(os.path.join(homeuser, 'blue01.npy'))
    blue02 = np.load(os.path.join(homeuser, 'blue02.npy'))
    blue03 = np.load(os.path.join(homeuser, 'blue03.npy'))

    # set up the various data configurations
    test_data_full = [blue00, blue01, blue02, blue03]
    header = ['E1', 'E2', 'G12', 'OBJ', 'Success']

    my_full = invbubble.BubbleOpt('my_full_test.csv', header,
                                  100.0, None, None,
                                  test_data=test_data_full,
                                  mat_model='lin-ortho')
    results = np.zeros(3)
    x_full = [0.26422968, 0.24657734, 0.25798352]

    results[0] = my_full.calc_obj_function_test_data(x_full, run_abq=True)
    write_lin_orth_model(x_full, random=False)
    invbubble.run_model()
    invbubble.export_csv_files()
    results[1] = my_full.calc_obj_function_test_data(x_full, run_abq=False)
    write_lin_orth_model(x_full, random=True)
    invbubble.run_model()
    invbubble.export_csv_files()
    results[2] = my_full.calc_obj_function_test_data(x_full, run_abq=False)
    print(results)
