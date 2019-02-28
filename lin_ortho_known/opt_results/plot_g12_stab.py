import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

g12res = pd.read_csv('g12stability.csv')
minobj = g12res.values[:, 4].min()
plt.figure()
isfail = 0
issuc = 0
for i in g12res.values:
    if i[-1] == 0:
        if isfail == 0:
            plt.semilogx(i[3]/100, minobj, 'xk', label='FEM failed')
            isfail = 1
        else:
            plt.semilogx(i[3]/100, minobj, 'xk')
    else:
        if issuc == 0:
            plt.semilogx(i[3]/100, i[4], 'ob', label='FEM Success')
            issuc = 1
        else:
            plt.semilogx(i[3]/100, i[4], 'ob')
plt.xlabel('G12 (GPa)')
plt.ylabel('Objective function (mm)')
plt.title(r'$E_1=0.26$ $E_2=0.24$ (GPa)')
plt.legend()
plt.grid()
plt.savefig('G12_line_plot.png', bbox_inches='tight')
plt.show()