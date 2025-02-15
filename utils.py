import numpy as np



# ----------------- Homogeneous Transform Utilities -----------------
def homogeneous_transform(DH_table_row):
    # Extract the DH parameters.
    a     = DH_table_row[0]
    alpha = DH_table_row[1]
    d     = DH_table_row[2]
    theta = DH_table_row[3]

    # Create the individual transformation matrices.
    Rot_z = Rot("z", theta)
    Tra_z = Tra("z", d)
    Tra_x = Tra("x", a)
    Rot_x = Rot("x", alpha)
    
    # Combine them: A = Rz(theta) * Tz(d) * Tx(a) * Rx(alpha)
    A = Rot_z @ Tra_z @ Tra_x @ Rot_x
    return A

def Rot(axis, angle):
    """
    Construct a 4x4 rotation matrix about a given axis.
    """
    R = np.eye(4)
    c = np.cos(angle)
    s = np.sin(angle)

    if axis == "x":
        R[1, 1] = c;  R[1, 2] = -s
        R[2, 1] = s;  R[2, 2] = c
    elif axis == "y":
        R[0, 0] = c;  R[0, 2] = s
        R[2, 0] = -s; R[2, 2] = c
    elif axis == "z":
        R[0, 0] = c;  R[0, 1] = -s
        R[1, 0] = s;  R[1, 1] = c
    else:
        print("[ERROR] Rot : Invalid axis. Returning identity.")
    return R

def Tra(axis, displacement):
    """
    Construct a 4x4 translation matrix along a given axis.
    """
    T = np.eye(4)
    if axis == "x":
        T[0, 3] = displacement
    elif axis == "y":
        T[1, 3] = displacement
    elif axis == "z":
        T[2, 3] = displacement
    else:
        print("[ERROR] Tra : Invalid axis. Returning identity.")
    return T

# ----------------- Polynomial order 3 Trajectory Design Utilities ---
def generate_trajectory(waypoints, velocities, position_limits, max_velocity, tf, T):
    """
    Given a list of waypoints and a list of velocities to pass over each
    waypoint and the total time tf to perform this motion with time interval 
    of T is computes the successive positions and composes the trajectory.
    :param waypoints : list of waypoints (each a 3-element list (x,y,z))
    :param velocities: list of velocities (each a 3-element list (vx,vy,vz))
    :param tf        : total time for the motion
    :return          : list of waypoints and the velocities at each point
    """
    is_valid = True # If false then something is out of range!

    x_lim, y_lim, z_lim = position_limits

    # Log trajectory points and velocity for each point
    trajectory   = []
    velocity     = []
    acceleration = []

    N = round(tf / T) # The number of timesteps to generate

    k = 0

    # iterate through all waypoints
    for k in range(len(waypoints)-1):
        x0, y0, z0 = waypoints[k]    # starting point on the trajectory
        x1, y1, z1 = waypoints[k+1]  # final point on the trajectory

        vx0, vy0, vz0 = velocities[k]    # initial velocity
        vx1, vy1, vz1 = velocities[k+1]  # final velocity

        # Find the cubic polynomial coefficients
        a0x, a1x, a2x, a3x = poly3_interpollation(x0, vx0, x1, vx1, tf)
        a0y, a1y, a2y, a3y = poly3_interpollation(y0, vy0, y1, vy1, tf)
        a0z, a1z, a2z, a3z = poly3_interpollation(z0, vz0, z1, vz1, tf)
        
        # compute the points per time-step
        for n in range(N):
            x = evaluate_poly3(a0x, a1x, a2x, a3x, n * T)
            y = evaluate_poly3(a0y, a1y, a2y, a3y, n * T)
            z = evaluate_poly3(a0z, a1z, a2z, a3z, n * T)
            
            # Check if the positions computed are within the workspace of the robot
            if x <= x_lim[0] or x >= x_lim[1]:
                print(f"[INFO] x is out of range: {x}")
                is_valid = False
                break
            
            if y <= y_lim[0] or y >= y_lim[1]:
                print(f"[INFO] y is out of range: {y}")
                is_valid = False
                break
                
            if z <= z_lim[0] or z >= z_lim[1]:
                print(f"[INFO] z is out of range: {z}")
                is_valid = False
                break

            trajectory.append(np.array([x, y, z]))     

        # compute the velocities per time step
        for n in range(len(trajectory)-1):
            xi, yi, zi = trajectory[n]
            xf, yf, zf = trajectory[n+1]

            vx = (xf - xi) / T
            vy = (yf - yi) / T
            vz = (zf - zi) / T
            
            mag_v = np.sqrt(vx**2 + vy**2 + vz**2)
            if mag_v > max_velocity:
                print(f"[INFO] velocity is out of bound: ({vx}, {vy}, {vz}), |v| = {mag_v}")
                is_valid = False
                 
            v = np.array([vx, vy, vz])

            velocity.append(v)

        # compute the acceleration per time step
        for n in range(len(velocity)-1):
            vxi, vyi, vzi = velocity[n]
            vxf, vyf, vzf = velocity[n+1]

            ax = (vxf - vxi) / T
            ay = (vyf - vyi) / T
            az = (vzf - vzi) / T
            
            a = np.array([ax, ay, az])

            acceleration.append(a)

    return trajectory, velocity, acceleration, is_valid

def poly3_interpollation(p0, v0, pf, vf, tf):
    """
    Compute the coefficients of a 3rd order polynomial interpolation
    :param p0 : initial position (single coordinate)
    :param v0 : initial velocity (single speed value)
    :param pf : final position (single coordinate)
    :param vf : final velocity (single speed value)
    :param tf : total time for the motion
    returns the computed coefficients of the polynomial
    """
    a0 = p0
    a1 = v0
    a2 = (3/tf**2) * (pf - p0) - (2/tf) * v0 - (1/tf) * vf 
    a3 = -(2/tf**3) * (pf - p0) + (1/tf**2) * (v0 + vf)
    return [a0, a1, a2, a3]

def evaluate_poly3(a0, a1, a2, a3, t):
    """
    Evaluates the 3-rd order polynomial with coefficients
    a0, a1, a2, a3 at time instant t. Returns the evaluated value.
    """
    return a0 + a1 * t + a2 * t**2 + a3 * t**3
