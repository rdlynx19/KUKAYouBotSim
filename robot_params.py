import numpy as np

# chassis frame to base frame of arm
Tb0 = np.array([[1, 0, 0, 0.1662], [0, 1, 0, 0], [0, 0, 1, 0.0026], [0, 0, 0, 1]])

# home configuration of end effector when all joint angles are zero
M0e = np.array([[1, 0, 0, 0.033], [0, 1, 0, 0],[0, 0, 1, 0.6546]])
B1 = np.array([0, 0, 1, 0, 0.033, 0])
B2 = np.array([0, -1, 0, -0.5076, 0, 0])
B3 = np.array([0, -1, 0, -0.3526, 0, 0])
B4 = np.array([0, -1, 0, -0.2176, 0, 0])
B5 = np.array([0, 0, 1, 0, 0, 0])

