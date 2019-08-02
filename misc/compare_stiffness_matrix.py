import numpy as np


# GPa
E = 0.198
G = 0.0689
nu = (E/(2.0*G)) - 1.0
E1 = 0.303
E2 = 0.229
G12 = 0.005
E23 = E2/2.48
# g12 = 1e-4
M = np.array([E1, E2, E2, 0.24, 0.24, 0.24, G12, G12, E23])

mat = {'E1': M[0], 'E2': M[1], 'E3': M[2], 'nu12': M[3], 'nu13': M[4],
       'nu23': M[5], 'G12': M[6], 'G13': M[7], 'G23': M[8]}
# other constants
mat['nu21'] = (mat['nu12']/mat['E1'])*mat['E2']
mat['nu31'] = (mat['nu13']/mat['E1'])*mat['E3']
mat['nu32'] = (mat['nu23']/mat['E2'])*mat['E3']

S_ortho = np.zeros((6, 6))
S_ortho[0, 0] = 1.0 / mat['E1']
S_ortho[0, 1] = - mat['nu21'] / mat['E2']
S_ortho[0, 2] = - mat['nu31'] / mat['E3']
S_ortho[1, 1] = 1.0 / mat['E2']
S_ortho[1, 0] = - mat['nu12'] / mat['E1']
S_ortho[1, 2] = - mat['nu32'] / mat['E3']
S_ortho[2, 2] = 1.0 / mat['E3']
S_ortho[2, 0] = - mat['nu13'] / mat['E1']
S_ortho[2, 1] = - mat['nu23'] / mat['E2']
S_ortho[3, 3] = 1.0 / mat['G12']
S_ortho[4, 4] = 1.0 / mat['G13']
S_ortho[5, 5] = 1.0 / mat['G23']
C_ortho = np.linalg.inv(S_ortho)
C_ortho = np.linalg.inv(S_ortho)
print(np.linalg.cond(S_ortho))
print(np.linalg.cond(C_ortho))
u, v = np.linalg.eig(S_ortho)
print(np.sort(u))

C_lin = np.eye(6)
C_lin[0:3] *= 1.0 - nu
C_lin[3:6] *= 1.0 - (2.0*nu)
C_lin[1, 0] = nu
C_lin[0, 1] = nu
C_lin[0, 2] = nu
C_lin[2, 0] = nu
C_lin[1, 2] = nu
C_lin[2, 1] = nu
C_lin *= E / ((1+nu)*(1-(2.0*nu)))

C_lin_3 = C_lin[0:3, 0:3]
C_ortho_3 = C_ortho[0:3, 0:3]
print('Name', 'Trace', 'Tensordot')
print('C_lin_3', np.trace(C_lin_3), np.tensordot(C_lin_3, C_lin_3))
print('C_ortho_3', np.trace(C_ortho_3), np.tensordot(C_ortho_3, C_ortho_3))