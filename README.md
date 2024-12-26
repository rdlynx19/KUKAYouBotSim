# ME449 Final Project: Mobile Manipulation with KUKA YouBot
## Milestone2: Reference Trajectory Generation
The file `trajectory_generation.py` generates a reference trajectory for the end effector from it's initial configuration to the standoff configuration after placing the cube at it's goal position.

It consists of two functions: `TrajectoryGenerator` which generates the reference trajectory in 8 segments, indicated in the docstring at the beginning of the file, and a function `populate_configuration_list` is used to format the robot configuration to write it correctly to the csv file.


## Milestone1: youBot Kinematics Simulator
The file `kinematics.py` simulates the kinematics of the YouBot. 

It consists of three functions: `odometry` which calculates the new chassis configuration using odometry equations, `limit_speeds` which clips the robot wheels and manipulator joint speeds within the specified limits, `NextState` which takes in the current configuration of the robot and outputs the next configuration after time dt.

## Milestone3: Feedforward Control
The file `feedforward_control.py` is used to perform feedback control of the mobile manipulator. 

It consists of two functions: `BaseandArmSpeeds` which computes the wheel and joint speeds from the end effector twist, `FeedbackControl` which performs feedforward and PI control to output the commanded end effector twist

# Complete Program
There are three different files based on the specified task.

The file `simulation_best.py` runs the program with the best tuned Kp and Ki values.

The file `simulation_overshoot.py` runs the program with the overshoot Kp and Ki values.

The file `simulation_newTask.py` runs the program with the newTask Kp and Ki values.

All these file consist of a `main` function which simulates the entire task and returns a list of the robot configurations and the error variation over time. The function `plot_error` plots the 6 vector error using matplotlib. The robot configurations are saved in a csv file using `np.savetxt`.

All the robot parameters are defined in a file `robot_params.py`. The timestep for all simulations `dt` is 0.01