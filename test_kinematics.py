"""Test run for the NextState function."""
import numpy as np
from kinematics import NextState

current_config = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
wheel_joint_speeds = np.array([-10, 10, 10, -10, 5, 0, 0, 5, 0])

config_list = []

for i in range(0, 100):
    config_row = []
    updated_config = NextState(current_config=current_config, wheel_joint_speeds=wheel_joint_speeds, dt=0.01, speed_limit=10.0)
    current_config = updated_config
    for elem in updated_config:
        config_row.append(elem)
    config_row.append(0.0)
    config_list.append(config_row)

np.savetxt('states.csv', config_list, delimiter=',')
    
    

