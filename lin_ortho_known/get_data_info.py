import numpy as np

blue00 = np.load('blue00.npy')
blue01 = np.load('blue01.npy')
blue02 = np.load('blue02.npy')
blue03 = np.load('blue03.npy')

test_data = [blue00, blue01, blue02, blue03]

for i in test_data:
    data = []
    for j in i:
        data.append(j[0].shape[0])
        # print(j[0].shape)
    data = np.array(data)
    print(data.sum())