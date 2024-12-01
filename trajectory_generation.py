"""
Generate the reference trajectory of the end-effector in 8 segments.

1.A trajectory to move the gripper from its initial configuration to a "standoff" configuration a few cm above the block.
2.A trajectory to move the gripper down to the grasp position.
3.Closing of the gripper.
4.A trajectory to move the gripper back up to the "standoff" configuration.
5.A trajectory to move the gripper to a "standoff" configuration above the final configuration.
6.A trajectory to move the gripper to the final configuration of the object.
7.Opening of the gripper.
8.A trajectory to move the gripper back to the "standoff" configuration.
"""
import numpy as np
import modern_robotics as mr

def populate_configuration_list(
        configuration_list, traj, gripper_state: int = 0):
    """
    :param traj: The trajectory of the segment to be added to the configuration list.
    :return configuration_list: The populated list of the end effector configurations
    """
    for i in range(0,len(traj)):
        configuration_row = []
        for k in range(0, 3):
            for m in range(0, 3):
                elem = traj[i][k,m]
                configuration_row.append(elem)
        configuration_row.append(traj[i][0,3])
        configuration_row.append(traj[i][1,3])
        configuration_row.append(traj[i][2,3])
        configuration_row.append(gripper_state)
        configuration_list.append(configuration_row)

    return configuration_list


def TrajectoryGenerator(Tse_initial, Tsc_initial, Tsc_final, Tce_grasp, Tce_standoff, k = 1):
    """
    :param Tse_initial: The initial configuration of the end effector in the reference trajectory.
    :param Tsc_initial: The cube's initial configuration
    :param Tsc_final: The cube's desired configuration
    :param Tce_grasp: The end effector's configuration relative to the cube when it is grasping the cube
    :param Tce_standoff: The end effector's configuration above the cube, before and after grasping
    :param k: The number of trajectory reference configurations per 0.01s
    :return Tse: A representation of the N configurations of the end effector along the entire concatenated 8 segment reference trajectory
    """
    configuration_list = []
    Tse_standoff = np.matmul(Tsc_initial, Tce_standoff)
    traj1 = mr.ScrewTrajectory(Tse_initial, Tse_standoff, 4, 400, 3)
    configuration_list = populate_configuration_list(configuration_list, traj1)

    Tse_grasp = np.matmul(Tsc_initial, Tce_grasp)
    
    traj2 = mr.CartesianTrajectory(Tse_standoff, Tse_grasp, 2, 200, 3)
    configuration_list = populate_configuration_list(configuration_list, traj2)

    traj3 = mr.CartesianTrajectory(Tse_grasp, Tse_grasp, 1, 100, 3)
    configuration_list = populate_configuration_list(configuration_list, traj3, 1)

    traj4 = mr.CartesianTrajectory(Tse_grasp, Tse_standoff, 2, 200, 3)
    configuration_list = populate_configuration_list(configuration_list, traj4, 1)

    Tse_final_standoff = np.matmul(Tsc_final, Tce_standoff)
    traj5 = mr.ScrewTrajectory(Tse_standoff, Tse_final_standoff, 6, 600, 3)
    configuration_list = populate_configuration_list(configuration_list, traj5, 1)

    Tse_final_grasp = np.matmul(Tsc_final, Tce_grasp)
    traj6 = mr.CartesianTrajectory(Tse_final_standoff, Tse_final_grasp, 2, 200, 3)
    configuration_list = populate_configuration_list(configuration_list, traj6, 1)

    traj7 = mr.CartesianTrajectory(Tse_final_grasp, Tse_final_grasp, 1, 100, 3)
    configuration_list = populate_configuration_list(configuration_list, traj7)

    traj8 = mr.ScrewTrajectory(Tse_final_grasp, Tse_final_standoff, 2, 200, 3)
    configuration_list = populate_configuration_list(configuration_list, traj8)
    return configuration_list



# Uncomment this section to test this function individually

# Tb0 = np.array([[1, 0, 0, 0.1662], [0, 1, 0, 0], [0, 0, 1, 0.0026], [0, 0, 0, 1]])
# M0e = np.array([[1, 0, 0, 0.033], [0, 1, 0, 0], [0, 0, 1, 0.6546], [0, 0, 0, 1]])
# Tsb = np.array([[1, 0, 0, 0], [0, 1, 0 , 0], [0, 0, 1, 0.0963], [0, 0, 0, 1]])
# Tse_initial = np.matmul(np.matmul(Tsb, Tb0), M0e)
# Tsc_initial = np.array([[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0.025], [0, 0, 0, 1]])
# Tsc_final = np.array([[0, 1, 0, 0], [-1, 0, 0, -1], [0, 0, 1, 0.025], [0, 0, 0, 1]])
# Tce_standoff = np.array([[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0 ,0, 0.35], [0, 0, 0, 1]]) # standoff configuration of the end effector
# Tce_grasp = np.array([[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0.015], [0, 0, 0, 1]])

# configuration_list = TrajectoryGenerator(Tse_initial=Tse_initial, Tsc_initial=Tsc_initial, Tsc_final=Tsc_final, Tce_grasp=Tce_grasp, Tce_standoff=Tce_standoff)
# np.savetxt("generated_trajectory.csv", configuration_list, delimiter=',')
