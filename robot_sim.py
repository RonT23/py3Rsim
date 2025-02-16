#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import utils
import user

class robot3RSim():
    def __init__(self, P_home=None, ax3d=None):
        """
        :param P_home: Home position for inverse kinematics (optional).
        :param ax3d  : A Matplotlib Axes3D object into which the robot will be drawn
                       (if None, a new figure and 3D axis are created)
        """
        # If the user did not defined a home point
        # use as default the initial configuration with
        # all joints set to zero in the theretical analysis
        if P_home is not None:
            self.P_home = P_home
        else:
            self.P_home = np.array([0.0, 0.0, 0.0])
        
        # set limits for joint movement (in radians)
        self.q1_range = (-np.pi, np.pi)
        self.q2_range = (-np.pi, np.pi)
        self.q3_range = (-np.pi, np.pi)

        # rad/ s
        self.dq1_max = 3
        self.dq2_max = 3
        self.dq3_max = 3
        
        # set the base at origin
        self.base = np.array([0.0, 0.0, 0.0])

        # The manipulator geometrical properties (in meters)
        # these variables are private!
        self.__l1 = 0.42   # distance from base to joint 1
        self.__l2 = 0.45   # length of link 1 (joint 1 to joint 3)
        self.__l3 = 0.835  # length of link 2 (joint 3 to end effector)

        # Initialize the robot to the home position/configuration
        self.Q = self.inverse_kinematics(self.P_home)
        
        # intialize the robot with joints at rest
        self.dQ = np.array([0.0, 0.0, 0.0]) # rad/sec

        # set workspace limits
        self.x_range = (-self.__l2 - self.__l3, self.__l2 + self.__l3)
        self.y_range = (-self.__l2 - self.__l3, self.__l2 + self.__l3)
        self.z_range = (0.0, self.__l1 + self.__l2 + self.__l3)

        # If no axis was provided, create a new figure and axis
        if ax3d is None:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111, projection='3d')
        else:
            self.ax = ax3d
            self.fig = ax3d.figure  # The figure associated with the provided 3D axis
        
        # Create line objects for the links 
        self.link_bj1,  = self.ax.plot([], [], [], 'o-', lw=6, color='b')
        self.link_j1j2, = self.ax.plot([], [], [], 'o-', lw=5, color='g')
        self.link_j2j3, = self.ax.plot([], [], [], 'o-', lw=4, color='r')
        self.link_j3ee, = self.ax.plot([], [], [], 'o-', lw=3, color='m')

        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')
        self.ax.set_xlim(-1, 1)
        self.ax.set_ylim(-1, 1)
        self.ax.set_zlim(0, 1.5)

    def DH_table(self, Q=None):
        """
        Construct the DH matrix based on the theoretical analysis.
        :param Q : the configuration vector [q1, q2, q3]
        returns the DH table matrix
        """
        if Q is not None:
            q1, q2, q3 = Q
        else:
            q1, q2, q3 = self.Q
         
        DH = np.array([
           #      a (m),    α (rad),        d (m),           θ (rad)   
            [   0.0,      np.pi/2,         self.__l1,          q1],            # base -> frame 1
            [self.__l2,        0.0,            0.0,     np.pi/2 + q2],           # frame 1 -> frame 2
            [   0.0,      np.pi/2,            0.0,     np.pi/2 + q3],           # frame 2 -> frame 3
            [   0.0,         0.0,           self.__l3,         0.0]                # frame 3 -> end effector
        ])

        return DH 

    def Jacobian_matrix(self, Q=None):
        """
        Compute the Jacobian matrix based on the theoretical analysis.
        :param Q: the configuration vector [q1, q2, q3]
        returns  the Jacobian matrix for linear velocities JL and angular velocities JA. 
        """
        if Q is not None:
            q1, q2, q3  = Q
        else:
            q1, q2, q3 = self.Q

        s1, c1      = np.sin(q1), np.cos(q1)
        s2, c2      = np.sin(q2), np.cos(q2)
        s23, c23    = np.sin(q2+q3), np.cos(q2+q3)

        # The linear velocity Jacobian
        JL = np.array([
            [-self.__l3 * s1 * s23 + self.__l2 * s1 * s2, -self.__l3 * c1 * c23 - self.__l2 * c1 * c2, -self.__l3 * c1 * c23],
            [-self.__l3 * c1 * s23 - self.__l2 * c1 * s2, -self.__l3 * s1 * c23 - self.__l2 * s1 * c2, -self.__l3 * s1 * c23],
            [            0.0                        , -self.__l3 * s23 - self.__l2 * s2          , -self.__l3 * s23     ]
        ])

        # The angular velocity Jacobian.
        # Note: Here this is not used!
        JA = np.array([
            [ 0.0,  s1,  s1],
            [ 0.0, -c1, -c1],
            [ 1.0, 0.0, 0.0]
        ])

        return JL, JA

    def forward_kinematics(self, Q=None):
        """
        Given a configuration [q1, q2, q3] for the 3R manipulator,
        return a 5x3 array where each row is the (x, y, z) position of:
          row 0: base,
          row 1: joint 1,
          row 2: joint 2,
          row 3: joint 3,
          row 4: end effector.
        """
        # update the DH table with the given configuration.
        # If None, use the internal state
        if Q is None:
            DH = self.DH_table()
        else:
            DH = self.DH_table(Q)
        
        # Compute the homogeneous local transform matrices
        A01 = utils.homogeneous_transform(DH[0])
        A12 = utils.homogeneous_transform(DH[1])
        A23 = utils.homogeneous_transform(DH[2])
        A3E = utils.homogeneous_transform(DH[3])
        
        # Compute the transforms from base to each subsequent frame
        T0_1 = A01                  # base -> frame 1
        T0_2 = T0_1 @ A12           # base -> frame 2
        T0_3 = T0_2 @ A23           # base -> frame 3
        T0_E = T0_3 @ A3E           # base -> end effector

        # Extract the Cartesian positions of each frame
        pos = np.zeros((5, 3))
        pos[0, :] = np.array(self.base) # base
        pos[1, :] = T0_1[0:3, 3]        # joint 1
        pos[2, :] = T0_2[0:3, 3]        # joint 2
        pos[3, :] = T0_3[0:3, 3]        # joint 3
        pos[4, :] = T0_E[0:3, 3]        # end effector

        return pos

    def inverse_kinematics(self, P):
        """
        Given the desired end-effector position P = (px, py, pz) (in meters),
        solve for the robot's joint angles [q1, q2, q3] (in rad).
        returns the joint configuration vector (3x1).
        """
        px, py, pz = P

        # Uesed for readability
        l0 = self.__l1    # offser along z
        l1 = self.__l2    # link 2 length
        l2 = self.__l3    # link 3 length

        # Compute the first joint angle from the x and y position
        q1 = np.arctan2(py, px)

        if q1 == np.pi/2 or q1 == -np.pi/2:
            q1 = q1 + (q1/abs(q1)) * np.deg2rad(1) # add to the direction of change 0.1 deggrees to avoid the zero

        # Convert the 3D problem into a 2D problem by performing a coordinate transformation
        x = px / np.cos(q1)  # horizontal distance in the plane
        z = pz - l0          # vertical distance
        r = x**2 + z**2 

        # Compute the elbow angle
        cos_q3 =  (r - l1**2 - l2**2) / (2 * l1 * l2)
        sin_q3 = np.sqrt(1 - cos_q3**2)
        q3     = -np.atan2(sin_q3, cos_q3)

        # Compute the shoulder angle
        sin_q2 = (x * (l1 + l2 * cos_q3) - z * l2 * sin_q3) / (l1**2 + l2**2 + 2 * l1 * l2 * cos_q3)
        cos_q2 = np.sqrt(1 - sin_q2**2)
        q2     = -np.atan2(sin_q2, cos_q2)

        Q = np.array([q1, q2, q3])

        # Note: I am using "-" for q3 and q2 because of the sign convention of the frames. 
        #       If q2 and q3 are used with "+" the robot will perform the motion from the opposite
        #       direction (like a mirror) to the actual positions set to track.

        return Q

    def forward_differential_kinematics(self, Q=None, dQ=None):
        """
        Solve the forward differential kinematics using the defined
        Jacobian matrix as analysed in the theoretical part.
        :param Q  : the configuration (q1, q2, q3) of the manipulator
        :param dQ : the joint velocities
        returns the linear and angular velocity vectors of the end effector
        """
        # Construct the Jacobian matrix for the current configuration
        if Q is not None:
            JL, JA  = self.Jacobian_matrix(Q)
        else: # use state
            JL, JA  = self.Jacobian_matrix()

        if dQ is not None:     
            v = JL @ dQ # Compute the 3x1 linear velocities vector
            w = JA @ dQ # Compute the 3x1 angular velocities vector
        else: # use state
            v = JL @ self.dQ # Compute the 3x1 linear velocities vector
            w = JA @ self.dQ # Compute the 3x1 angular velocities vector

        return v, w
    
    def generate_signals(self, P, dt):
        q1_out, q2_out, q3_out      = [], [], []    # store the joint displacement
        dq1_out, dq2_out, dq3_out   = [], [], []    # store the joint velocities
        
        for i, point in enumerate(P):

            # compute the joint displacement
            q1, q2, q3 = self.inverse_kinematics(point)
            
            # Compute the joint speed
            dq1 = (self.Q[0] - q1)/dt
            dq2 = (self.Q[1] - q2)/dt
            dq3 = (self.Q[2] - q3)/dt

            # update the displacement state
            self.update_q1(q1)
            self.update_q2(q2)
            self.update_q3(q3)

            # update the velocity state
            self.update_dq1(dq1)
            self.update_dq2(dq2)
            self.update_dq3(dq3)

            # Log data       
            q1_out.append(q1)
            q2_out.append(q2)
            q3_out.append(q3)
            dq1_out.append(dq1)
            dq2_out.append(dq2)
            dq3_out.append(dq3)

        # Before returning the values re-initialize the state to the initial one
        # so we can use it for the simulation later
        self.dQ = np.array([0.0, 0.0, 0.0])
        self.Q  = self.inverse_kinematics(self.P_home)

        return q1_out, q2_out, q3_out, dq1_out, dq2_out, dq3_out
            
    # ---------------- Main simulation function -------------------------
    def run(self, q1, q2, q3, dq1, dq2, dq3):
        """
        For each configuration vector and velocity vector update the internal joint configuration,
        compute the forward kinematics and end effector velocities with differential kinematics and
        animate the robot motion.
        :param q1, q2, q3    : the vector of displacement signals per time instant
        :param dq1, dq2, dq3 : the vector of joint speed signals per time instant
        returns the end effector position and the end effector linear velocities per timestep.
        """
        s = [] # store the position data for all frames of reference
               # this is used for the animation logic

        px_out, py_out, pz_out      = [], [], []    # store the end effector position
        vx_out, vy_out, vz_out      = [], [], []    # store the end effector velocities

        for i in range(len(q1)):

            # Update the robot's internal joint state (position and velocity)            
            self.update_q1(q1[i])
            self.update_q2(q2[i])
            self.update_q3(q3[i])
            self.update_dq1(dq1[i])
            self.update_dq2(dq2[i])
            self.update_dq3(dq3[i])

            # Compute the end effector velocities. If the velocity of 
            # each joint is grater than the maximum then use the maximum.
            # we should see the difference if any at the graphs!
            # We do not care for the angular velocities, only linear!
            v, _ = self.forward_differential_kinematics()

            # Compute the positions of each frame for that configuration
            # this is used for the animation logic
            pos = self.forward_kinematics()

            # Log data
            vx_out.append(v[0])
            vy_out.append(v[1])
            vz_out.append(v[2])
            
            px_out.append(pos[4][0])
            py_out.append(pos[4][1])
            pz_out.append(pos[4][2])
            
            s.append(pos)
        
        # Animate the motion of the robot using the precomputed configurations
        self.animate_robot_motion(s)

        return px_out, py_out, pz_out, vx_out, vy_out, vz_out
    
    # ----------------- Internal state update functions -----------------
    def update_q1(self, q1):
        if q1 < self.q1_range[0] or q1 > self.q1_range[1]:
            print("[WARNING] q1 is out of range")
        else:
            self.Q[0] = q1

    def update_q2(self, q2):
        if q2 < self.q2_range[0] or q2 > self.q2_range[1]:
            print("[WARNING] q2 is out of range")
        else:
            self.Q[1] = q2

    def update_q3(self, q3):
        if q3 < self.q3_range[0] or q3 > self.q3_range[1]:
            print("[WARNING] q3 is out of range")
        elif q3 == 0 or q3 == -np.pi:  
            print("[WARNING] q3 reached a singularity. Adjusting value.")
            self.Q[2] = q3 + np.deg2rad(0.1)
        else:
            self.Q[2] = q3

    def update_dq1(self, dq1):
        if abs(dq1) > self.dq1_max:
            self.dQ[0] = (dq1 / abs(dq1)) * self.dq1_max # bound to maximum permited
            print("[WARNING] dq1 is out of range")
        else:
            self.dQ[0] = dq1

    def update_dq2(self, dq2):
        if abs(dq2) > self.dq2_max:
            self.dQ[1] = (dq2 / abs(dq2)) * self.dq2_max
            print("[WARNING] dq2 is out of range")
        else:
            self.dQ[1] = dq2
    
    def update_dq3(self, dq3):
        if abs(dq3) > self.dq3_max:
            self.dQ[2] = (dq3 / abs(dq3)) * self.dq3_max
            print("[WARNING] dq3 is out of range")
        else:
            self.dQ[2] = dq3

    # ----------------- Visualization Utilities -----------------
    def add_link(self, link, P1, P2):
        x1, y1, z1 = P1
        x2, y2, z2 = P2
        link.set_data_3d([x1, x2], [y1, y2], [z1, z2])
 
    def animate_robot_motion(self, all_positions):
        """
        Animate the robot motion by drawing the links between the frames.
        param all_positions: list of 5x3 arrays (for each configuration).
        """
        # Animate the robot motion along the positions computed
        for pos in all_positions:
            # Extract positions of each frame
            pB  = pos[0]  # Base
            pJ1 = pos[1]  # Joint 1
            pJ2 = pos[2]  # Joint 2
            pJ3 = pos[3]  # Joint 3
            pEE = pos[4]  # End-Effector

            # Set the links between frames
            self.add_link(self.link_bj1, pB, pJ1)
            self.add_link(self.link_j1j2, pJ1, pJ2)
            self.add_link(self.link_j2j3, pJ2, pJ3)
            self.add_link(self.link_j3ee, pJ3, pEE)

            # Mark the end-effector and via point
            self.ax.plot([pEE[0]], [pEE[1]], [pEE[2]], 'k.', markersize=2, alpha=0.8)
           
            plt.draw()
            plt.pause(0.05)

def main():

    # read the user configuration for the simulation
    waypoints, velocities, accelerations, tf, max_v, max_a = user.task()
    
    dt = 0.01  # time resolution, i.e. the sampling period from point to point

    # Creta the figures for the animation and graphs
    fig_animation = plt.figure(figsize=(12, 10))
    ax3d = fig_animation.add_subplot(111, projection='3d')
    
    # Create the robot homed at the first waypoint
    robot = robot3RSim(waypoints[0], ax3d)

    # Get the robot-specific limits that define its workspace
    # and create the limits vector. Intrinsic to the robot constraints.
    # However the paramters are updatable!
    position_limis = [robot.x_range, 
                      robot.y_range, 
                      robot.z_range]

    # Plot the waypoints to the figure
    for point in waypoints:
        x, y, z = point
        ax3d.plot([x], [y], [z], 'ro', markersize=10, alpha=0.8)

    # Generate a smooth trajectory using 5-th order polynomial interpolation
    trajectory, velocity, acceleration, is_valid = utils.generate_trajectory_5th_roder(waypoints, velocities, accelerations, position_limis, max_v, max_a, tf, dt)
    
    # in case the generated trajectory has violated constraints 
    # then don't start the simulation, because it will not work!
    if(not is_valid):
        print("[ERROR] Invalid position!")
        exit(-1)
    
    ####### SIMULATION CALL HERE ################################################

    # generate the signals
    q1, q2, q3, dq1, dq2, dq3 = robot.generate_signals(trajectory, dt)    
    
    # run the simualtion of the robot
    px, py, pz, vx, vy, vz = robot.run(q1, q2, q3, dq1, dq2, dq3)
    
    #############################################################################
    
    # local buffers for the target data (generated from the generate_trajectory_5th_roder() function)
    tx, ty, tz    = [], [], []
    vtx, vty, vtz = [], [], []
    atx, aty, atz = [], [], []

    # fill these buffers with actual data
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

    # Build a time vector based on the values of the simulated joints
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
    ax21.plot(time_vector[0:len(tx)], tx[0:len(tx)], "r", label="X")
    ax21.plot(time_vector[0:len(tx)], ty[0:len(tx)], "b", label="Y")
    ax21.plot(time_vector[0:len(tx)], tz[0:len(tx)], "g", label="Z")
    ax21.grid(True)
    ax21.legend()

    # Plot 6: Target Velocity Profile
    ax22.set_xlabel('Time (sec)')
    ax22.set_ylabel('Velocity (m/s)')
    ax22.set_title("Target Linear Velocity Profile")
    ax22.plot(time_vector[0:len(vtx)], vtx[0:len(vtx)], "r", label="X")
    ax22.plot(time_vector[0:len(vtx)], vty[0:len(vtx)], "b", label="Y")
    ax22.plot(time_vector[0:len(vtx)], vtz[0:len(vtx)], "g", label="Z")
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