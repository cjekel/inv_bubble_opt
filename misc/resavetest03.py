import numpy as np
import os
homeuser = os.path.expanduser('~')

blue03 = np.load(os.path.join(homeuser, 'blue03.npy'), allow_pickle=True)
# blue03 = blue03[18:]
# np.save('blue03.npy', blue03)
