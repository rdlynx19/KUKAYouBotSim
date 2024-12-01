import numpy as np
import modern_robotics as mr
import matplotlib.pyplot as plt
from trajectory_generation import TrajectoryGenerator
from kinematics import NextState
from feedforward_control import FeedbackControl, BaseandArmSpeeds

Tb0 = np.array([[1, 0, 0, 0.1662], [0, 1, 0, 0], [0, 0, 1, 0.0026], [0, 0, 0, 1]])

M0e = np.array([[1, 0, 0, 0.033], [0, 1, 0, 0], [0, 0, 1, 0.6546], [0, 0, 0, 1]])

Blist = np.array([[0, 0, 1, 0, 0.033, 0],
                       [0, -1, 0, -0.5076, 0, 0],
                       [0, -1, 0, -0.3526, 0, 0],
                       [0, -1, 0, -0.2176, 0, 0],
                       [0, 0, 1, 0, 0, 0]]).T

Tsb = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.0963], [0, 0, 0, 1]])

# Tse_initial = np.matmul(np.matmul(Tsb, Tb0), M0e)
Tse_initial = np.array([[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0.5], [0, 0, 0, 1]])

Tsc_initial = np.array([[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0.025], [0, 0, 0, 1]])
Tsc_final = np.array([[0, 1, 0, 0], [-1, 0, 0, -1], [0, 0, 1, 0.025], [0, 0, 0, 1]])

Tce_standoff = np.array(
    [[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0.35], [0, 0, 0, 1]]
)  # standoff configuration of the end effector

Tce_grasp = np.array([[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0.005], [0, 0, 0, 1]])

dt = 0.01
k = 1

Kp = 30 * np.eye(6)
Ki = 1 * np.eye(6)

Xerr_tot = 0.0

robot_configuration = [0.1, 0.2, 0.1, 0, 0, 0.2, -1.6, 0, 0, 0, 0, 0, 0]


reference_trajectory = TrajectoryGenerator(
    Tse_initial, Tsc_initial, Tsc_final, Tce_grasp, Tce_standoff, k=1
)
error_variation = []
simulated_configurations = []
simulated_configurations.append(robot_configuration)
for i in range(0, len(reference_trajectory) - 1):

    Xd = np.vstack(
        (
            reference_trajectory[i][0:3],
            reference_trajectory[i][3:6],
            reference_trajectory[i][6:9],
        )
    )
    Xd = np.hstack((Xd, np.reshape(reference_trajectory[i][9:12], (3, 1))))
    Xd = np.vstack((Xd, np.array([0, 0, 0, 1])))

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
    X = np.matmul(Tsb, Tbe)

    V, Xerr = FeedbackControl(X, Xd, Xd_nxt, Kp, Ki, dt, Xerr_tot)
    wheel_joint_speeds = BaseandArmSpeeds(V, thetalist)

    robot_configuration = NextState(
        robot_configuration[0:12], wheel_joint_speeds, dt)
    robot_configuration = np.append(
        robot_configuration, reference_trajectory[i][-1])
    simulated_configurations.append(robot_configuration)
    error_variation.append(Xerr)

np.savetxt("full_run.csv", simulated_configurations, delimiter=",")
print(len(error_variation))
error_variation.append(np.array([0, 0, 0, 0, 0, 0]))
t = np.arange(0,1999*dt, dt)
print(t.shape)
error_variation = np.array(error_variation)
for i in range(6):
    plt.plot(t, error_variation[:, i], label=f'Component {i+1}')

# Customize the plot
plt.title('6-Vector Plot Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Value')
plt.legend()
plt.grid(True)

# Display the plot
plt.show()



