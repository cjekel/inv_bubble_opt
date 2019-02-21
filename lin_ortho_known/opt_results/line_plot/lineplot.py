import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

mydf = pd.read_csv('my_line_search.csv')

alp = np.arange(0, 100)

plt.figure()
plt.plot(alp, mydf.values[:, 4], '.-')
plt.xlabel('alpha')
plt.ylabel('Objective value (mm)')
plt.grid()
plt.show()
