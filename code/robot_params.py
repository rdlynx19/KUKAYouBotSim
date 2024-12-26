import numpy as np

# The fixed offset from the chassis frame {b} to the base frame of the arm {0}
Tb0 = np.array([[1, 0, 0, 0.1662], 
                [0, 1, 0, 0], 
                [0, 0, 1, 0.0026], 
                [0, 0, 0, 1]])

# The end effector frame {e} relative to the base frame {0}, when the arm is at its home configuration
M0e = np.array([[1, 0, 0, 0.033], 
                [0, 1, 0, 0], 
                [0, 0, 1, 0.6546], 
                [0, 0, 0, 1]])

# The screw axes for the 5 arm joints, expressed in the end effector frame (when the arm is at its home configuration)
Blist = np.array([[0, 0, 1, 0, 0.033, 0],
                [0, -1, 0, -0.5076, 0, 0],
                [0, -1, 0, -0.3526, 0, 0],
                [0, -1, 0, -0.2176, 0, 0],
                [0, 0, 1, 0, 0, 0]]).T


# Tse_initial = np.matmul(np.matmul(Tsb, Tb0), M0e)
# The initial configuration of the end effector reference trajectory
Tse_initial = np.array([[0, 0, 1, 0], 
                        [0, 1, 0, 0], 
                        [-1, 0, 0, 0.5], 
                        [0, 0, 0, 1]])

# The initial configuration of the cube
Tsc_initial = np.array([[1, 0, 0, 1], 
                        [0, 1, 0, 0], 
                        [0, 0, 1, 0.025], 
                        [0, 0, 0, 1]])

# The final configuration of the cube
Tsc_final = np.array([[0, 1, 0, 0], 
                      [-1, 0, 0, -1], 
                      [0, 0, 1, 0.025], 
                      [0, 0, 0, 1]])

# The initial configuration of the cube for a newTask
Tsc_initial_newTask = np.array([[1, 0, 0, 1.5],
                                [0, 1, 0, 0],
                                [0, 0, 1, 0.025],
                                [0, 0, 0, 1]])

# The final configuration of the cube for a newTask
Tsc_final_newTask = np.array([[1/np.sqrt(2), 1/np.sqrt(2), 0, 0.5],
                              [-1/np.sqrt(2), 1/np.sqrt(2), 0, -1],
                              [0, 0, 1, 0.025],
                              [0, 0, 0, 1]])

# The standoff configuration of the end effector, before and after grasping the cube
Tce_standoff = np.array([[0, 0, 1, 0], 
                         [0, 1, 0, 0], 
                         [-1, 0, 0, 0.35], 
                         [0, 0, 0, 1]])  

# The configuration of the end effector when it grasps the cube
Tce_grasp = np.array([[0, 0, 1, 0], 
                      [0, 1, 0, 0], 
                      [-1, 0, 0, 0.005], 
                      [0, 0, 0, 1]])

# The configuration of frame {b} of the mobile base, relative to frame {s} on the floor. Here, (x,y,phi) = (0, 0, 0)
# Tsb = np.array([[1, 0, 0, 0], 
#                 [0, 1, 0, 0], 
#                 [0, 0, 1, 0.0963], 
#                 [0, 0, 0, 1]])