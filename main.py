import numpy as np
import matplotlib.pyplot as plt
import utils
from robot_sim import *

def main():
    ###### USER DEFINITIONS BELLOW ###############

    tf    = 4    # time per linear segment
    dt    = 0.01 # time resolution
    max_v = 4.0  # maximum permited velocity magnitude (m/s)
    max_a = 10.0  # maximum permited acceleration magnitude (m/s^2)

    # User defined waypoints and velocities
    waypoints = np.array([( 0.45, 0.1,  1.3),
                          ( 0.45, 0.45, 0.01), 
                          ( 0.45, 0.1,  1.3)
                        ])
    velocities = np.array([(0, 0, 0), 
                           (0, 0, 0), 
                           (0, 0, 0)])
    accelerations = np.array([(0, 0, 0), 
                              (0, 0, 0), 
                              (0, 0, 0)])
    ##############################################
    
    ####### DONT TOUCH THESE #####################

    # Creta the figures for the animation and graphs
    fig_animation = plt.figure(figsize=(12, 10))
    ax3d = fig_animation.add_subplot(111, projection='3d')
    
    # Create the robot homed at the first waypoint
    robot = robot3RSim(waypoints[0], ax3d)

    # Get the robot-specific limits that define its workspace
    x_lim = robot.x_range
    y_lim = robot.y_range
    z_lim = robot.z_range

    # create the limits vector
    position_limis = [x_lim, 
                      y_lim, 
                      z_lim]

    # Plot the waypoints 
    for point in waypoints:
        x, y, z = point
        ax3d.plot([x], [y], [z], 'ro', markersize=10, alpha=0.8)

    # Generate a smooth trajectory using 3-rd order polynomial interpollation
    #trajectory, velocity, acceleration, is_valid = utils.generate_trajectory_3rd_order(waypoints, velocities, position_limis, max_v, max_a, tf, dt)

    # Generate a smooth trajectory using 5-th order polynomial interpolation
    trajectory, velocity, acceleration, is_valid = utils.generate_trajectory_5th_roder(waypoints, velocities, accelerations, position_limis, max_v, max_a, tf, dt)
    
    if(not is_valid):
        print("[ERROR] Invalid position!")
        exit(-1)
    
    tx, ty, tz    = [], [], []
    vtx, vty, vtz = [], [], []
    atx, aty, atz = [], [], []

    for position in trajectory:
        tx.append(position[0])
        ty.append(position[1])
        tz.append(position[2])

    for vel in velocity:
        vtx.append(vel[0])
        vty.append(vel[1])
        vtz.append(vel[2])

    for acc in acceleration:
        atx.append(acc[0])
        aty.append(acc[1])
        atz.append(acc[2])

    # Run the simulation
    q1, q2, q3, px, py, pz, dq1, dq2, dq3, vx, vy, vz = robot.run(trajectory, dt)

    # Build a time vector
    time_vector = np.arange(len(q1)) * dt

    _, axs1 = plt.subplots(2, 2, figsize=(12, 8))
    (ax11, ax12), (ax13, ax14) = axs1

    # Plot 1: End Effector Trajectory
    ax11.set_xlabel('Time (sec)')
    ax11.set_ylabel('Position (m)')
    ax11.set_title("End Effector Trajectory")
    ax11.plot(time_vector, px, "r", label="X")
    ax11.plot(time_vector, py, "b", label="Y")
    ax11.plot(time_vector, pz, "g", label="Z")
    ax11.grid(True)
    ax11.legend()

    # Plot 2: End Effector Linear Velocity
    ax12.set_xlabel('Time (sec)')
    ax12.set_ylabel('Velocity (m/s)')
    ax12.set_title("End Effector Linear Velocity")
    ax12.plot(time_vector, vx, "r", label="X")
    ax12.plot(time_vector, vy, "b", label="Y")
    ax12.plot(time_vector, vz, "g", label="Z")
    ax12.grid(True)
    ax12.legend()

    # Plot 3: Joint Trajectories
    ax13.set_xlabel('Time (sec)')
    ax13.set_ylabel('Displacement (deg)')
    ax13.set_title("Joint Displacement")
    ax13.plot(time_vector, np.rad2deg(q1), "r", label="q1")
    ax13.plot(time_vector, np.rad2deg(q2), "g", label="q2")
    ax13.plot(time_vector, np.rad2deg(q3), "b", label="q3")
    ax13.grid(True)
    ax13.legend()

    # Plot 4: Joint Velocities 
    ax14.set_xlabel('Time (sec)')
    ax14.set_ylabel('Velocity (deg/sec)')
    ax14.set_title("Joint Velocities")
    ax14.plot(time_vector, np.rad2deg(dq1), "r", label="q1")
    ax14.plot(time_vector, np.rad2deg(dq2), "g", label="q2")
    ax14.plot(time_vector, np.rad2deg(dq3), "b", label="q3")
    ax14.grid(True)
    ax14.legend()

    _, axs2 = plt.subplots(3, 1, figsize=(12, 8))
    (ax21, ax22, ax23) = axs2

    # Plot 5: Target Trajectory Profile
    ax21.set_xlabel('Time (sec)')
    ax21.set_ylabel('Position (m)')
    ax21.set_title("Target Trajectory Profile")
    ax21.plot(time_vector[0:len(atx)], tx[0:len(atx)], "r", label="X")
    ax21.plot(time_vector[0:len(atx)], ty[0:len(atx)], "b", label="Y")
    ax21.plot(time_vector[0:len(atx)], tz[0:len(atx)], "g", label="Z")
    ax21.grid(True)
    ax21.legend()

    # Plot 6: Target Velocity Profile
    ax22.set_xlabel('Time (sec)')
    ax22.set_ylabel('Velocity (m/s)')
    ax22.set_title("Target Linear Velocity Profile")
    ax22.plot(time_vector[0:len(atx)], vtx[0:len(atx)], "r", label="X")
    ax22.plot(time_vector[0:len(atx)], vty[0:len(atx)], "b", label="Y")
    ax22.plot(time_vector[0:len(atx)], vtz[0:len(atx)], "g", label="Z")
    ax22.grid(True)
    ax22.legend()

    # Plot 7: Target Acceleration Profile
    ax23.set_xlabel('Time (sec)')
    ax23.set_ylabel('Acceleration (m^2/s)')
    ax23.set_title("Target Linear Acceleration Profile")
    ax23.plot(time_vector[0:len(atx)], atx[0:len(atx)], "r", label="X")
    ax23.plot(time_vector[0:len(atx)], aty[0:len(atx)], "b", label="Y")
    ax23.plot(time_vector[0:len(atx)], atz[0:len(atx)], "g", label="Z")
    ax23.grid(True)
    ax23.legend()

    plt.tight_layout()
    plt.show()

    print("[INFO] Simulation Complete!")
    
if __name__ == '__main__':
    main()
