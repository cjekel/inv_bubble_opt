import numpy as np
import pandas as pd

res = np.load('blue_cross_compute_res.npy')

header = ['Full OBJ', 'CV1 OBJ', 'CV2 OBJ', 'CV3 OBJ', 'CV4 OBJ']
header2 = ['Full mat', 'CV1 mat', 'CV2 mat', 'CV3 mat', 'CV4 mat']

mydf = pd.DataFrame(res, index=header2, columns=header)
