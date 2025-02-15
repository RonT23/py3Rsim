#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import utils

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
       
        # set the base at origin
        self.base = np.array([0.0, 0.0, 0.0])

        # The manipulator geometrical properties (in meters)
        self.l1 = 0.42   # distance from base to joint 1
        self.l2 = 0.45   # length of link 1 (joint 1 to joint 3)
        self.l3 = 0.835  # length of link 2 (joint 3 to end effector)

        # Initialize the robot to the home position/configuration
        self.Q = self.inverse_kinematics(self.P_home)

        # set workspace limits
        self.x_range = (-self.l2 - self.l3, self.l2 + self.l3)
        self.y_range = (-self.l2 - self.l3, self.l2 + self.l3)
        self.z_range = (0.0, self.l1 + self.l2 + self.l3)

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

    def DH_table(self, Q):
        """
        Construct the DH matrix based on the theoretical analysis.
        :param Q : the configuration vector [q1, q2, q3]
        returns the DH table matrix
        """
        q1, q2, q3 = Q

        DH = np.array([
           #      a (m),    α (rad),        d (m),           θ (rad)   
            [   0.0,      np.pi/2,         self.l1,          q1],            # base -> frame 1
            [self.l2,        0.0,            0.0,     np.pi/2 + q2],           # frame 1 -> frame 2
            [   0.0,      np.pi/2,            0.0,     np.pi/2 + q3],           # frame 2 -> frame 3
            [   0.0,         0.0,           self.l3,         0.0]                # frame 3 -> end effector
        ])

        return DH 

    def Jacobian_matrix(self, Q):
        """
        Compute the Jacobian matrix based on the theoretical analysis.
        :param Q: the configuration vector [q1, q2, q3]
        returns  the Jacobian matrix for linear velocities JL and angular velocities JA. 
        """
        q1, q2, q3  = Q

        s1, c1      = np.sin(q1), np.cos(q1)
        s2, c2      = np.sin(q2), np.cos(q2)
        s23, c23    = np.sin(q2+q3), np.cos(q2+q3)

        # The linear velocity Jacobian
        JL = np.array([
            [-self.l3 * s1 * s23 + self.l2 * s1 * s2, -self.l3 * c1 * c23 - self.l2 * c1 * c2, -self.l3 * c1 * c23],
            [-self.l3 * c1 * s23 - self.l2 * c1 * s2, -self.l3 * s1 * c23 - self.l2 * s1 * c2, -self.l3 * s1 * c23],
            [            0.0                        , -self.l3 * s23 - self.l2 * s2          , -self.l3 * s23     ]
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
        if Q==None:
            DH = self.DH_table(self.Q)
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
        l0 = self.l1    # offser along z
        l1 = self.l2    # link 2 length
        l2 = self.l3    # link 3 length

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

        # Note: I am using "-" for q3 and q2 because of the sign convention. If q2 and q3 
        #       are used with "+" the robot will perform the motion from the opposite
        #       direction (like a mirror) to the actual positions set to track.

        return Q

    def forward_differential_kinematics(self, Q, dQ):
        """
        Solve the forward differential kinematics using the defined
        Jacobian matrix as analysed in the theoretical part.
        :param Q  : the configuration (q1, q2, q3) of the manipulator
        :param dQ : the joint velocities
        returns the linear and angular velocity vectors of the end effector
        """
        # Construct the Jacobian matrix for the current configuration
        JL, JA  = self.Jacobian_matrix(Q)

        v = JL @ dQ # Compute the 3x1 linear velocities vector
        w = JA @ dQ # Compute the 3x1 angular velocities vector

        return v, w
    
    # ---------------- Main simulation function -------------------------
    def run(self, P, dt):
        """
        This is the simulations main functionality. For each waypoint in P, compute the corresponding 
        joint angles using inverse kinematics, update the internal joint configuration, compute forward
        kinematics, compute velocities of joints and end effector and animate the robot motion.
        :param P  : the list of target waypoints as generated by a trajectory generator.
        :param dt : the time interval used to compute the time derivatives
        returns the joint displacement, the end effector position, the joint velocities and
                the end effector linear velocities per timestep.
        """
        s = [] # store the position data for all frames of reference
               # this is used for the animation logic

        q1, q2, q3      = [], [], []    # store the joint displacement
        px, py, pz      = [], [], []    # store the end effector position
        dq1, dq2, dq3   = [], [], []    # store the joint velocities
        vx, vy, vz      = [], [], []    # store the end effector velocities

        for point in P:
            # Compute the joint angles for the target point using inverse kinematics
            try:
                Q = self.inverse_kinematics(point)
            except ValueError as e:
                # In case we cannot compute the inverse kinematics skip this specific point
                print(f"Skipping point {point}: {e}")
                continue
            
            # Compute the velocities of joint displacement and log data
                # (current value - state value) / time interval 
            dq1.append( (Q[0] - self.Q[0])/dt )
            dq2.append( (Q[1] - self.Q[1])/dt )
            dq3.append( (Q[2] - self.Q[2])/dt )
            
            # Update the robot's internal joint state
            self.update_q1(Q[0])
            self.update_q2(Q[1])
            self.update_q3(Q[2])

            # Compute the end effector velocities
                # we do not care for the angular velocities!
            dQ = np.array([dq1[-1], dq2[-1], dq3[-1]])
            v, _ = self.forward_differential_kinematics(self.Q, dQ)

            # Compute the positions of each frame for that configuration
            pos = self.forward_kinematics()

            # Log data
            q1.append(Q[0])
            q2.append(Q[1])
            q3.append(Q[2])

            vx.append(v[0])
            vy.append(v[1])
            vz.append(v[2])
            
            px.append(pos[4][0])
            py.append(pos[4][1])
            pz.append(pos[4][2])

            s.append(pos)

        # Visualize the motion using the computed configurations
        self.animate_robot_motion(s)

        return q1, q2, q3, px, py, pz, dq1, dq2, dq3, vx, vy, vz
    
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
    ###### USER DEFINITIONS BELLOW ###############

    tf    = 1    # time per linear segment
    dt    = 0.01 # time resolution
    max_v = 3.0  # maximum permited velocity magnitude (m/s)

    # User defined waypoints and velocities
    waypoints = np.array([( 0.45, 0.1,  1.3),
                          ( 1.0, 0.45,  0.5), 
                          ( 0.45, 0.45, 0.01)
                        ])
    velocities = np.array([(0, 0, 0), 
                           (0, 0, -1), 
                           (0, 0, 0)])
    
    ##############################################
    
    ####### DONT TOUCH THESE #####################

    # Creta the figures for the animation and graphs
    fig_animation = plt.figure(figsize=(12, 10))
    ax3d = fig_animation.add_subplot(111, projection='3d')
    
    # Create the robot homed at the first waypoint
    robot = robot3RSim(waypoints[0], ax3d)

    x_lim = robot.x_range
    y_lim = robot.y_range
    z_lim = robot.z_range

    position_limis = [x_lim, y_lim, z_lim]

    # Optionally, plot the waypoints in the same 3D axis
    for point in waypoints:
        x, y, z = point
        ax3d.plot([x], [y], [z], 'ro', markersize=10, alpha=0.8)

    # Generate a smooth trajectory
    trajectory, _, _, is_valid = utils.generate_trajectory(waypoints, velocities, position_limis, max_v, tf, dt)
    if(not is_valid):
        print("[ERROR] Invalid position!")
        exit(-1)

    # Run the simulation
    q1, q2, q3, px, py, pz, dq1, dq2, dq3, vx, vy, vz = robot.run(trajectory, dt)

    # Build a time vector
    time_vector = np.arange(len(q1)) * dt

    _, axs = plt.subplots(2, 2, figsize=(12, 8))
    (ax1, ax2), (ax3, ax4) = axs

    # Plot 1: End Effector Trajectory
    ax1.set_xlabel('Time (sec)')
    ax1.set_ylabel('Position (m)')
    ax1.set_title("End Effector Trajectory")
    ax1.plot(time_vector, px, "r", label="X")
    ax1.plot(time_vector, py, "b", label="Y")
    ax1.plot(time_vector, pz, "g", label="Z")
    ax1.grid(True)
    ax1.legend()

    # Plot 2: End Effector Linear Velocity
    ax2.set_xlabel('Time (sec)')
    ax2.set_ylabel('Velocity (m/s)')
    ax2.set_title("End Effector Linear Velocity")
    ax2.plot(time_vector, vx, "r", label="X")
    ax2.plot(time_vector, vy, "b", label="Y")
    ax2.plot(time_vector, vz, "g", label="Z")
    ax2.grid(True)
    ax2.legend()

    # Plot 3: Joint Trajectories
    ax3.set_xlabel('Time (sec)')
    ax3.set_ylabel('Displacement (deg)')
    ax3.set_title("Joint Displacement")
    ax3.plot(time_vector, np.rad2deg(q1), "r", label="q1")
    ax3.plot(time_vector, np.rad2deg(q2), "g", label="q2")
    ax3.plot(time_vector, np.rad2deg(q3), "b", label="q3")
    ax3.grid(True)
    ax3.legend()

    # Plot 4: Joint Velocities 
    ax4.set_xlabel('Time (sec)')
    ax4.set_ylabel('Speed (deg/sec)')
    ax4.set_title("Joint Velocities")
    ax4.plot(time_vector, np.rad2deg(dq1), "r", label="q1")
    ax4.plot(time_vector, np.rad2deg(dq2), "g", label="q2")
    ax4.plot(time_vector, np.rad2deg(dq3), "b", label="q3")
    ax4.grid(True)
    ax4.legend()

    plt.tight_layout()
    plt.show()

    print("[INFO] Simulation Complete!")
    
if __name__ == '__main__':
    main()
