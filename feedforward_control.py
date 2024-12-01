"""Kinematic task space feedforward plus feedforward control law."""
import modern_robotics as mr
import numpy as np
np.set_printoptions(suppress=True)

def BaseandArmSpeeds(end_effector_twist, thetalist):
    """
    Compute the wheel and joint speeds from the end effector twist

    :param end_effector_twist: The end-effector twist in its own frame
    """
    M0e = np.array([[1, 0, 0, 0.033], [0, 1, 0, 0], [0, 0, 1, 0.6546], [0, 0, 0, 1]])
    Tb0 = np.array([[1, 0, 0, 0.1662], [0, 1, 0, 0], [0, 0, 1, 0.0026], [0, 0, 0, 1]])
    Blist = np.array([[0, 0, 1, 0, 0.033, 0],
                       [0, -1, 0, -0.5076, 0, 0],
                       [0, -1, 0, -0.3526, 0, 0],
                       [0, -1, 0, -0.2176, 0, 0],
                       [0, 0, 1, 0, 0, 0]]).T
    
    r = 0.0475  # radius of the wheels
    l = 0.47/2  # half of forward backward distance between wheels
    w = 0.3/2  

    Jarm = mr.JacobianBody(Blist, thetalist)
    F6 = (r/4) * np.array([[0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [-1/(l+w), 1/(l+w), 1/(l+w), -1/(l+w)], 
                        [1, 1, 1, 1], 
                        [-1, 1, -1, 1],
                        [0, 0, 0, 0]])
    T0e = mr.FKinBody(M0e, Blist, thetalist)
    T0e_inv = mr.TransInv(T0e)
    # adTe0_T0b = mr.Adjoint(np.matmul(T0e_inv, mr.TransInv(Tb0)))
    adTe0_T0b = mr.Adjoint(np.matmul(T0e_inv, Tb0))
    Jbase = np.dot(adTe0_T0b,F6)
    Jrob = np.hstack((Jbase, Jarm))
    # print(f'J: {Jrob}')
    wheel_and_joints = np.matmul(np.linalg.pinv(Jrob, 1e-4), end_effector_twist)
    # print(f'wheel and joint speeds: {wheel_and_joints}')
    return wheel_and_joints


def FeedbackControl(X, Xd, Xd_next, Kp, Ki, dt, Xerr_tot):
    """
    :param X: current actual end effector configuration
    :param Xd: current end effector reference configuration
    :param Xd_next: end effector reference configuration at the next timestep in the reference trajectory (after time dt)
    :param Kp: proportional gain matrix
    :param Ki: integral gain matrix
    :param dt: timestep between reference trajectories
    """
    Xerr = mr.se3ToVec(mr.MatrixLog6(np.matmul(mr.TransInv(X), Xd)))
    Xerr_tot = Xerr_tot + (Xerr * dt)
    Vd_mat = (1/dt)*(mr.MatrixLog6(np.matmul(mr.TransInv(Xd), Xd_next)))
    Vd = mr.se3ToVec(Vd_mat)
    adX_Xd = mr.Adjoint(np.matmul(mr.TransInv(X), Xd))
    V = (np.matmul(adX_Xd, Vd)) + (np.dot(Kp, Xerr)) + (np.dot(Ki, Xerr_tot))
    # print(f'Vd: {Vd}')        
    # print(f'AdVd: {np.matmul(adX_Xd, Vd)}')
    # print(f'V: {V}')
    # print(f'Xerr: {Xerr}')
    # print(f'Jb: {Jrob}')
    # print(f'speed_list: {speed_lists}')
    return V, Xerr

# V, Xerr = FeedbackControl(X, Xd, Xd_next, Kp, Ki, dt, Xerr_tot)
# BaseandArmSpeeds(V)