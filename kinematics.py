"""Simulator for the kinematics of the YouBot."""
import numpy as np
import modern_robotics as mr


def odometry(chassis_config, wheel_speeds, dt):
    """
    Calculate the new chassis configuration using odometry

    :param chassis_config: current configuration of the chassis
    :param wheel_speeds: speed of the 4 wheels
    :param dt: timestep dt
    """
    r = 0.0475  # radius of the wheels
    l = 0.47/2  # half of forward backward distance between wheels
    w = 0.3/2   # half of side to side distance between wheels
    F = (r/4) * np.array([
        [-1/(l+w), 1/(l+w), 1/(l+w), -1/(l+w)], 
        [1, 1, 1, 1], 
        [-1, 1, -1, 1]])
    Vb = np.matmul(F, (wheel_speeds*dt))
    if Vb[0] == 0.0:
        delta_phib = 0.0
        delta_xb = Vb[1]
        delta_yb = Vb[2]
    else:
        delta_phib = Vb[0]
        delta_xb = (Vb[1]*np.sin(Vb[0]) + Vb[2]*(np.cos(Vb[0]) - 1))/Vb[0]
        delta_yb = (Vb[2]*np.sin(Vb[0]) + Vb[1]*(1 - np.cos(Vb[0])))/Vb[0]
    delta_qb = np.array([delta_phib, delta_xb, delta_yb])
    chassis_angle = chassis_config[0]
    T_sb = np.array([[1, 0, 0], 
                     [0, np.cos(chassis_angle), -np.sin(chassis_angle)], 
                     [0, np.sin(chassis_angle), np.cos(chassis_angle)]])
    delta_q = np.matmul(T_sb, delta_qb)
    new_chassis_config = chassis_config + delta_q

    return new_chassis_config

def limit_speeds(robot_speeds, speed_limit):
    """
    Clip the list of speeds within the speed_limit.

    :param robot_speeds: list of wheel or arm joint speeds
    :param speed_limit: float value indicating the speed limit
    :return robot_speeds: the clipped value of robot speeds
    """
    for i in range(0, robot_speeds.shape[0]):
        if(robot_speeds[i] > speed_limit):
            robot_speeds[i] = speed_limit
        if(robot_speeds[i] < -speed_limit):
            robot_speeds[i] = -speed_limit

    return robot_speeds

def NextState(current_config, wheel_joint_speeds, dt, speed_limit):
    """
    :param current_config: 12 vector representing current configuration of the robot (chassis, arm and wheel angles)
    :param wheel_joint_speeds: 9 vector representing the wheel and arm joints speeds
    :param dt: timestep dt
    :param speed_limit: range of wheel and joint speeds (-speed_limit to +speed_limit)
    :return configuration: 12 vector representing the robot configuration after time dt
    """
    chassis_config = current_config[0:3]
    arm_joint_angles = current_config[3:8]
    joint_speeds = limit_speeds(wheel_joint_speeds[4:9], speed_limit=speed_limit)
    new_arm_joint_angles = arm_joint_angles + (joint_speeds * dt)
    wheel_angles = current_config[8:12]
    wheel_speeds = limit_speeds(wheel_joint_speeds[0:4], speed_limit=speed_limit)
    new_wheel_angles = wheel_angles + (wheel_speeds * dt)
    new_chassis_config = odometry(chassis_config=chassis_config, wheel_speeds=wheel_speeds, dt=dt)
    new_robot_config = np.hstack((new_chassis_config, new_arm_joint_angles, new_wheel_angles))

    return new_robot_config


