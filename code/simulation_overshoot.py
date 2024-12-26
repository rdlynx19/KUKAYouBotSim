import numpy as np
import modern_robotics as mr
import matplotlib.pyplot as plt
from trajectory_generation import TrajectoryGenerator
from kinematics import NextState
from feedforward_control import FeedbackControl, BaseandArmSpeeds
import robot_params

# Controller Gains, P and I

# Case 1: Best Case
# Kp = 5 * np.eye(6)
# Ki = 4 * np.eye(6)

# Case 2: Overshoot Case
Kp = 1 * np.eye(6)
Ki = 1 * np.eye(6)

# # Case 3: New Task
# Kp = 10 * np.eye(6)
# Ki = 5 * np.eye(6)

# Keeping track of the cumulative error, for the controller integral term
tot_Xerr = 0.0

# The initial robot configuration
robot_configuration = [0.1, 0.2, 0.3, 0, 0, 0.2, -1.6, 0, 0, 0, 0, 0, 0]


def main(Tb0, M0e, Blist, Tse_initial, Tsc_initial, Tsc_final, 
         Tce_grasp, Tce_standoff, robot_configuration, dt, Kp, Ki):
    """
    The full program to simulate the YouBot Pick and Place Task.

    :param Tb0: Fixed offset from chassis frame {b} to base frame of arm
    :param M0e: End erffector frame relative to base, when arm is at home configuration
    :param Blist: Screw axes of the arm, expressed in end effector frame
    :param Tse_initial: Initial configuration of end effector reference trajectory
    :param Tsc_initial: Initial configuration of the cube
    :param Tsc_final: Final configuration of the cube
    :param Tce_grasp: Configuration of the end effector when it grasps the cube
    :param Tce_standoff: Configuration of the end effector before and after grasping the cube
    :param robot_configuration: Initial configuration of the robot
    :param dt: time step of simulation
    :param Kp: Proportional gain matrix
    :param Ki: Integral gain matrix
    """
    reference_trajectory = TrajectoryGenerator(Tse_initial, 
                                               Tsc_initial,Tsc_final, Tce_grasp, Tce_standoff, k=1)
    error_variation = [] 
    # Storing the robot configuration for each time step
    simulated_configurations = [] 
    simulated_configurations.append(robot_configuration)
    # Iterating over the reference trajectory 
    for i in range(0, len(reference_trajectory) - 1):
        # the current desired state
        Xd = np.vstack(
            (
                reference_trajectory[i][0:3],
                reference_trajectory[i][3:6],
                reference_trajectory[i][6:9],
            )
        )
        Xd = np.hstack((Xd, np.reshape(reference_trajectory[i][9:12], (3, 1))))
        Xd = np.vstack((Xd, np.array([0, 0, 0, 1])))
        # the next desired state
        Xd_nxt = np.vstack(
            (
                reference_trajectory[i + 1][0:3],
                reference_trajectory[i + 1][3:6],
                reference_trajectory[i + 1][6:9],
            )
        )
        Xd_nxt = np.hstack((Xd_nxt, np.reshape(
            reference_trajectory[i + 1][9:12], (3, 1))))
        Xd_nxt = np.vstack((Xd_nxt, np.array([0, 0, 0, 1])))

        
        thetalist = robot_configuration[3:8]
        phi, x, y = robot_configuration[0:3]
        Tsb = np.array(
            [
                [np.cos(phi), -np.sin(phi), 0, x],
                [np.sin(phi), np.cos(phi), 0, y],
                [0, 0, 1, 0.0963],
                [0, 0, 0, 1],
            ]
        )
        T0e = mr.FKinBody(M0e, Blist, thetalist)
        Tbe = np.matmul(Tb0, T0e)
        X = np.matmul(Tsb, Tbe) # the current robot state

        # Controlling the robot from current to desired state
        V, Xerr = FeedbackControl(X, Xd, Xd_nxt, Kp, Ki, dt, tot_Xerr)
        wheel_joint_speeds = BaseandArmSpeeds(V, thetalist)
        # Calculating the next state of the robot
        robot_configuration = NextState(
            robot_configuration[0:12], wheel_joint_speeds, dt)
        # Appending the gripper state to the robot configuration
        robot_configuration = np.append(
            robot_configuration, reference_trajectory[i][-1])
        # Appending the robot configuration and error values
        simulated_configurations.append(robot_configuration)
        error_variation.append(Xerr)

    return simulated_configurations, error_variation

def plot_error(error_variation):
    """
    :param error_variation: list of the values of the error vector 
    """
    t = np.linspace(0, 19.99, 1999)
    error_variation = np.array(error_variation)
    for i in range(6):
        plt.plot(t, error_variation[:, i], label=f'Component {i+1}')
    plt.title('Error Plot Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Error Values')
    plt.legend()
    plt.grid(True)
    plt.show()


simulated_configurations, error_variation = main(robot_params.Tb0, 
                                                robot_params.M0e,
                                                robot_params.Blist,robot_params.Tse_initial,
                                                robot_params.Tsc_initial,
                                                robot_params.Tsc_final,
                                                robot_params.Tce_grasp,
                                                robot_params.Tce_standoff,
                                                robot_configuration,
                                                dt = 0.01,
                                                Kp = Kp,
                                                Ki = Ki)
# Saving the list of robot configurations to a csv file
print("Generating animation csv file.")
np.savetxt("overshoot_run.csv", simulated_configurations, delimiter=",")
print("Displaying error plot.")
np.savetxt("overshoot_run_error.csv", error_variation, delimiter=",")
plot_error(error_variation) # Plotting the error
print("Done.")
