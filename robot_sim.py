# def quintic_coefficients(t0, tf, q0, qf, v0, vf, a0, af):
#     T = tf - t0
#     M = np.array([
#         [1,  0,  0,    0,    0,    0   ],
#         [0,  1,  0,    0,    0,    0   ],
#         [0,  0,  2,    0,    0,    0   ],
#         [1,  T,  T**2, T**3, T**4, T**5],
#         [0,  1,  2*T,  3*T**2,  4*T**3,   5*T**4 ],
#         [0,  0,  2,    6*T,   12*T**2,  20*T**3 ]
#     ])
#     b = np.array([q0, v0, a0, qf, vf, af], dtype=float)
#     a = np.linalg.solve(M, b)
#     return a

# def evaluate_quintic(coeffs, t0, t):
#     tau = t - t0
#     a0,a1,a2,a3,a4,a5 = coeffs
#     q  = a0 + a1*tau + a2*tau**2 + a3*tau**3 + a4*tau**4 + a5*tau**5
#     dq = a1 + 2*a2*tau + 3*a3*tau**2 + 4*a4*tau**3 + 5*a5*tau**4
#     ddq= 2*a2 + 6*a3*tau + 12*a4*tau**2 + 20*a5*tau**3
#     return q, dq, ddq

# def generate_quintic_joint_trajectory(q0, qf, total_time=2.0, dt=0.01):
#     """
#     Single 5th-order polynomial from q0->qf in 'total_time'.
#     Start/end at zero velocity & accel.
#     Returns: (t_array, q_array, dq_array)
#     """
#     t0 = 0.0
#     tf = total_time
#     coeffs = quintic_coefficients(t0, tf, q0, qf, 0,0,0,0)
#     t_array = np.arange(t0, tf+dt, dt)
#     q_vals  = np.zeros_like(t_array)
#     dq_vals = np.zeros_like(t_array)
#     for i, t in enumerate(t_array):
#         qv, dqi, _ = evaluate_quintic(coeffs, t0, t)
#         q_vals[i]  = qv
#         dq_vals[i] = dqi
#     return t_array, q_vals, dq_vals



        

#     def differential_kinematics(self, J, dQ):
#         dq1, dq2, dq3, dq4, dq5, dq6 = dQ

#         v = J[0:2, 0:2] @ dQ[0:2]
#         w = J[3:5, 0:2] @ dQ[0:2]


#     # Jacobian matrix for the 3 first joints keeping the wrist 
#     # at initial position. 
#     def Jacobian_matrix(self, Q):
#         q1, q2, q3, q4, q5, q6 = Q
#         s1, c1 = np.sin(q1), np.cos(q1)
#         s2, c2 = np.sin(q2), np.cos(q2)
#         s23, c23 = np.sin(q2 + q3), np.cos(q2 + q3)

#         J = np.array([
#             [-0.835 * s1 * s23 + 0.45 * s1 * s2, -0.835 * c1 * c23 - 0.45 * c1 * c2, -0.835 * c1 * c23],
#             [-0.835 * c1 * s23 - 0.45 * c1 * s2, -0.835 * s1 * c23 - 0.45 * s1 * c2, -0.835 * s1 * c23],
#             [0.0, -0.835 * s23 - 0.45 * s2, -0.835 * s23],
#             [0.0, s1, s1],
#             [0.0, -c1, -c1],
#             [0.0, 0.0, 0.0]
#         ])

#         return J




        
# def deg2rag_Q(Q_deg):
#     Q_rad = Q_deg

#     for i, q in enumerate(Q_deg):
#         Q_rad[i] = np.deg2rad(q)
    
#     return Q_rad
 
# if __name__ == "__main__":
    
#     T_total  = 10 # Total time for the task in seconds
#     Tb       = 1  # Sampling period in seconds  

#     # point = (x, y, z) 
#     waypoints = [
#         (0.0, 0.835 + 0.45, 0.42),
#         # (0.2, 0.4, 0.5),
#         # (0.2, 0.6, 0.5)
#     ]

#     # Create a robot instance
#     robot = robot3RSim()

#     P = robot.create_path_test_q1(180)

#     robot.run(P)





#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import utils
import test_utils

class robot3RSim():
    def __init__(self):
        # initialize the robot configuration variables
        self.q1 = 0.0
        self.q2 = 0.0
        self.q3 = 0.0

        # set limits for joint movement (radians)
        self.q1_range = (-np.pi, np.pi)          
        self.q2_range = (-np.pi, np.pi)  
        self.q3_range = (-np.pi, np.pi)  
       
        # set the base at origin
        self.base = np.array([0.0, 0.0, 0.0])

        # The manipulator geometrical properties (in meters)
        self.l1 = 0.42   # distance from base to joint 1
        self.l2 = 0.45   # length of link 1 (joint 1 to joint 3)
        self.l3 = 0.835  # length of link 2 (joint 3 to end effector)

        # Create the simulation graphical environment
        plt.ion()
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
 
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

    def DH_table(self, q1, q2, q3):
        """
        Construct the DH matrix based on the theoretical analysis.
        """
        DH = np.array([
           #      a (m),    α (rad),        d (m),           θ (rad)   
            [   0.0,      np.pi/2,         self.l1,          q1],            # base -> frame 1
            [self.l2,        0.0,            0.0,     np.pi/2 + q2],           # frame 1 -> frame 2
            [   0.0,      np.pi/2,            0.0,     np.pi/2 + q3],           # frame 2 -> frame 3
            [   0.0,         0.0,           self.l3,         0.0]                # frame 3 -> end effector
        ])
        return DH 

    def forward_kinematics(self, q1=None, q2=None, q3=None):
        """
        Given a configuration Q = [q1, q2, q3] for the 3R manipulator,
        return a 5x3 array where each row is the (x, y, z) position of:
          row 0: base,
          row 1: joint 1,
          row 2: joint 2,
          row 3: joint 3,
          row 4: end effector.
        """
        # update the DH table with the given configuration
        if q1 == None or q2 == None or q3 == None:
            DH = self.DH_table(self.q1, self.q2, self.q3)
        else:
            DH = self.DH_table(q1, q2, q3)
        
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
        Given the desired end-effector position P = (px, py, pz) in meters,
        solve for the robot's joint angles Q = [q1, q2, q3] in radians.
        """
        px, py, pz = P

        # Uesed for readability
        a0 = self.l1    # offser along z
        a1 = self.l2    # link 2 length
        a2 = self.l3    # link 3 length

        # Compute the first joint angle from the x and y position
        q1 = np.arctan2(py, px)

        # Convert the 3D problem into a 2D problem by performing a coordinate transformation
        r = px / np.cos(q1)  # horizontal distance in the plane
        z = pz - a0          # vertical distance

        rho = r**2 + z**2 

        # Compute the elbow angle
        th = (rho - a1**2 - a2**2) / (2 * a1 * a2)
        if th < -1.0 or th > 1.0:
            # protect from exceptions
            raise ValueError("The desired position is unreachable.")
        else:
            q3 = np.arccos(th)

        # Compute the shoulder angle
        s3, c3 = np.sin(q3), np.cos(q3)
        q2 = np.atan2(r, z) - np.atan2(a2 * s3, a1 + a2 * c3)

        # Adjust the signs
        q2 = -q2 
        q3 = -q3
    
        return q1, q2, q3

    def run(self, P):
        """
        For each waypoint in P, compute the corresponding joint angles using inverse
        kinematics, update the internal joint configuration, compute forward kinematics,
        and animate the robot motion.
        """
        s = []  # list to store the forward kinematics positions for each configuration

        for point in P:
            # Compute the joint angles for the target point using inverse kinematics
            try:
                q1, q2, q3 = self.inverse_kinematics(point)
            except ValueError as e:
                # In case we cannot compute the inverse kinematics skip this point
                print(f"Skipping point {point}: {e}")
                continue

            # Update the robot's internal joint state
            self.update_q1(q1)
            self.update_q2(q2)
            self.update_q3(q3)

            # Compute the positions of each frame for that configuration
            pos = self.forward_kinematics()
            s.append(pos)

        # Visualize the motion using the computed configurations.
        self.animate_robot_motion(s)

    # ----------------- Internal state update functions -----------------
    def update_q1(self, q1):
        if q1 < self.q1_range[0] or q1 > self.q1_range[1]:
            print("[WARNING] q1 is out of range")
        else: 
            self.q1 = q1
    
    def update_q2(self, q2):
        if q2 < self.q2_range[0] or q2 > self.q2_range[1]:
            print("[WARNING] q2 is out of range")
        else: 
            self.q2 = q2

    def update_q3(self, q3):
        if q3 < self.q3_range[0] or q3 > self.q3_range[1]:
            print("[WARNING] q3 is out of range")
        else: 
            self.q3 = q3

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

        plt.ioff()
        plt.show()


# def quintic_coefficients(t0, tf, q0, qf, v0, vf, a0, af):
#     T = tf - t0
#     M = np.array([
#         [1,  0,  0,    0,    0,    0   ],
#         [0,  1,  0,    0,    0,    0   ],
#         [0,  0,  2,    0,    0,    0   ],
#         [1,  T,  T**2, T**3, T**4, T**5],
#         [0,  1,  2*T,  3*T**2,  4*T**3,   5*T**4 ],
#         [0,  0,  2,    6*T,   12*T**2,  20*T**3 ]
#     ])
#     b = np.array([q0, v0, a0, qf, vf, af], dtype=float)
#     a = np.linalg.solve(M, b)
#     return a

# def evaluate_quintic(coeffs, t0, t):
#     tau = t - t0
#     a0,a1,a2,a3,a4,a5 = coeffs
#     q  = a0 + a1*tau + a2*tau**2 + a3*tau**3 + a4*tau**4 + a5*tau**5
#     dq = a1 + 2*a2*tau + 3*a3*tau**2 + 4*a4*tau**3 + 5*a5*tau**4
#     ddq= 2*a2 + 6*a3*tau + 12*a4*tau**2 + 20*a5*tau**3
#     return q, dq, ddq

# def generate_quintic_joint_trajectory(q0, qf, total_time=2.0, dt=0.01):
#     """
#     Single 5th-order polynomial from q0->qf in 'total_time'.
#     Start/end at zero velocity & accel.
#     Returns: (t_array, q_array, dq_array)
#     """
#     t0 = 0.0
#     tf = total_time
#     coeffs = quintic_coefficients(t0, tf, q0, qf, 0,0,0,0)
#     t_array = np.arange(t0, tf+dt, dt)
#     q_vals  = np.zeros_like(t_array)
#     dq_vals = np.zeros_like(t_array)
#     for i, t in enumerate(t_array):
#         qv, dqi, _ = evaluate_quintic(coeffs, t0, t)
#         q_vals[i]  = qv
#         dq_vals[i] = dqi
#     return t_array, q_vals, dq_vals

# def multi_via_trajectory(end_effector_points, q0, time_per_segment=2.0, dt=0.01):
#     """
#     1) For each via-point, do IK => q_des
#     2) Generate a single quintic from current q->q_des
#     3) Concatenate times & joint angles
#     """
#     q_current = q0.copy()
#     T_offset = 0.0

#     T_all   = []
#     Q_all   = []
#     dQ_all  = []

#     for i, p_des in enumerate(end_effector_points):
#         # IK => next joint angles
#         q_des = inverse_kinematics(*p_des)

#         # We do a single-quintic for each joint, all with the same time_per_segment.
#         # We'll unify them by the same time array => pick the max length or just share it.
#         # Simpler approach: we assume each joint uses total_time = time_per_segment exactly.

#         n_steps = int(time_per_segment/dt)+1
#         q_temp  = np.zeros((n_steps, 3))
#         dq_temp = np.zeros((n_steps, 3))
#         t_local = np.linspace(0, time_per_segment, n_steps)
#         t_temp  = t_local + T_offset

#         for j in range(3):
#             # joint j
#             _, qj, dqj = generate_quintic_joint_trajectory(
#                 q_current[j], q_des[j],
#                 total_time=time_per_segment, dt=dt
#             )
#             q_temp[:, j]  = qj
#             dq_temp[:, j] = dqj

#         # Concatenate
#         if i == 0:
#             T_all = t_temp
#             Q_all = q_temp
#             dQ_all= dq_temp
#         else:
#             # remove first sample to avoid duplication
#             T_all = np.concatenate([T_all, t_temp[1:]])
#             Q_all = np.concatenate([Q_all, q_temp[1:]], axis=0)
#             dQ_all= np.concatenate([dQ_all, dq_temp[1:]], axis=0)

#         # update
#         T_offset = T_all[-1]
#         q_current = q_des

#     return T_all, Q_all, dQ_all


# Utility to generate a dense trajectory from sparse waypoints.
def generate_trajectory(waypoints, steps_per_segment=100):
    """
        Given a list of waypoints (each a [x, y, z] list), interpolate linearly
        between each successive pair to create a dense trajectory.
        :param waypoints: list of waypoints (each a 3-element list)
        :param steps_per_segment: number of interpolated points between each pair.
        :return: list of waypoints (including intermediate points)
    """
    trajectory = []
    for i in range(len(waypoints) - 1):
        start = np.array(waypoints[i])
        end   = np.array(waypoints[i+1])

        # Generate interpolated points (excluding the final point to avoid duplicates)
        for t in np.linspace(0, 1, steps_per_segment, endpoint=False):
            point = (1 - t) * start + t * end
            trajectory.append(point.tolist())

    # Append the final waypoint.
    trajectory.append(waypoints[-1])

    return trajectory


def main():
    # Create an instance of the robot simulation.
    robot = robot3RSim()

    # Time per segment
    T   = 2.0
    dt  = 0.01 # time resolution



    # Read the waypoints
    #waypoints = test_utils.create_circle(0.8, 0.2, 10)
    #waypoints = test_utils.create_letter_R()
    
    point_1 = (0.5, 0.3, 1.2)
    point_2 = (0.5, 0.3, 0.8)
    point_3 = (0.5, 0.3, 0.4)

    waypoints = np.array([point_1, 
                          point_2, 
                          point_3
                        ])

    # Display the waypoints
    for point in waypoints:
        x, y, z = point[0], point[1], point[2]
        robot.ax.plot([x], [y], [z], 'ro', markersize=10, alpha=0.8)
    
    # Generate a smooth trajectory by interpolating between the waypoints
    steps_per_segment = 20  # number of interpolation steps between waypoints
    trajectory = generate_trajectory(waypoints, steps_per_segment)

    time_per_segment = 2.0
    dt = 0.01

    # q0 = robot.inverse_kinematics(point_1)
    # T_all, Q_all, dQ_all = multi_via_trajectory(waypoints, q0, time_per_segment, dt)
    # print(f"Total trajectory duration: {T_all[-1]:.2f} s with {len(T_all)} steps")

    # Run the simulation on the trajectory
    robot.run(trajectory)

    print("[INFO] Simulation Complete!")
    
if __name__ == '__main__':
    main()
