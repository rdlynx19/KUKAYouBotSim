# ME449 Final Project: Mobile Manipulation with KUKA YouBot
## Milestone2: Reference Trajectory Generation
The file `trajectory_generation.py` generates a reference trajectory for the end effector from it's initial configuration to the standoff configuration after placing the cube at it's goal position.

It consists of two functions: `TrajectoryGenerator` which generates the reference trajectory in 8 segments, indicated in the docstring at the beginning of the file, and a function `populate_configuration_list` is used to format the robot configuration to write it correctly to the csv file.

To run the file, call the function `TrajectoryGenerator` with the appropriate parameters which are already defined in the file. To generate the csv file use the function `np.savetxt` and save the variable `configuration_list` to the csv filename of your choice. By default, running the code will generate a csv file called `generated_trajectory.csv` which can then be used in CoppeliaSim to visualise the trajectory.

