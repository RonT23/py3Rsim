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

# ----------------- 3-rd Order Polynomial Interpolation Utilities ----
def generate_trajectory_3rd_order(waypoints, velocities, position_limits, max_velocity, max_acceleration, tf, T):
    """
    Given a list of waypoints and a list of velocities to pass over each
    waypoint and the total time tf to perform this motion with time interval 
    of T is computes the successive positions and composes the trajectory.
    :param waypoints : list of waypoints (each a 3-element list (x,y,z))
    :param velocities: list of velocities (each a 3-element list (vx,vy,vz))
    :position_limits : list of a pair of limits for each coordinate ([min_x, max_x], [min_y, max_y], [min_z, max_z])
    :max_velocity    : the maximum velocity permited in means of magnitude (norm(V))
    :max_acceleration: the maximum acceleration permited in means of magnitude (norm(a))
    :param tf        : total time for the motion
    :return          : list of waypoints and the velocities at each point if valid.
    """
    is_valid = True # If false then something is out of range!

    x_lim, y_lim, z_lim = position_limits

    # Log trajectory points and velocity for each point
    trajectory_out   = []
    velocity_out     = []
    acceleration_out = []

    N = round(tf / T) # The number of timesteps to generate

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

        # compute the via points per time-step
        for n in range(0, N):
            x = evaluate_poly3(a0x, a1x, a2x, a3x, n * T)
            y = evaluate_poly3(a0y, a1y, a2y, a3y, n * T)
            z = evaluate_poly3(a0z, a1z, a2z, a3z, n * T)
            
            # Check if the positions computed are within the workspace of the robot
            if x <= x_lim[0] or x >= x_lim[1]:
                print(f"[WARNING] x is out of range: {x}")
                is_valid = False
                break
            
            if y <= y_lim[0] or y >= y_lim[1]:
                print(f"[WARNING] y is out of range: {y}")
                is_valid = False
                break
                
            if z <= z_lim[0] or z >= z_lim[1]:
                print(f"[WARNING] z is out of range: {z}")
                is_valid = False
                break

            trajectory_out.append(np.array([x, y, z]))     

        # Stop the first loop if the nedted ones are terminated with error
        if not is_valid:
            break

    # compute the velocities per time step
    for n in range(0, len(trajectory_out)-1):
        xi, yi, zi = trajectory_out[n]
        xf, yf, zf = trajectory_out[n+1]

        vx = (xf - xi) / T
        vy = (yf - yi) / T
        vz = (zf - zi) / T
        
        mag_v = np.sqrt(vz**2 + vy**2 + vz**2)
        if mag_v > max_velocity:
            print(f"[WARNING] velocity is out of bound: ({vx}, {vy}, {vz}), |v| = {mag_v}")
            is_valid = False
            break
 
        velocity_out.append(np.array([vx, vy, vz]))

    # compute the acceleration per time step
    for n in range(0, len(velocity_out)-1):
        vxi, vyi, vzi = velocity_out[n]
        vxf, vyf, vzf = velocity_out[n+1]

        ax = (vxf - vxi) / T
        ay = (vyf - vyi) / T
        az = (vzf - vzi) / T
        
        mag_a = np.sqrt(ax**2 + ay**2 + az**2)
        if mag_a > max_acceleration:
            print(f"[WARNING] acceleration is out of bound: ({ax}, {ay}, {az}), |a| = {mag_a}")
            is_valid = False
            break 

        acceleration_out.append(np.array([ax, ay, az]))

    return trajectory_out, velocity_out, acceleration_out, is_valid

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


# ----------------- 5-th Order Polynomial Interpolation Utilities -----

def poly5_interpolation(p0, v0, a0, pf, vf, af, tf):
    """
    Compute the coefficients of a 5th-order polynomial that satisfies:
    p(0)       = p0   ,   p(tf)       = pf
    dot p'(0)  = v0   ,   dot p(tf)   = vf
    ddot p(0)  = a0   ,   ddot p (tf) = af

    :param p0: initial position
    :param v0: initial velocity
    :param a0: initial acceleration
    :param pf: final position
    :param vf: final velocity
    :param af: final acceleration
    :param tf: total time of the segment

    :return: [b0, b1, b2, b3, b4, b5] so that p(t) = b0 + b1 t + b2 t^2 + b3 t^3 + b4 t^4 + b5 t^5
    """
    # Known direct placements:
    b0 = p0
    b1 = v0
    b2 = a0 / 2.0

    # We have three unknowns: b3, b4, b5
    # Derive them from the boundary conditions at t = tf for p, p', p''.
    # p(tf)   = b0 + b1*tf + b2*tf^2 + b3*tf^3 + b4*tf^4 + b5*tf^5 = pf
    # p'(tf)  = b1 + 2*b2*tf + 3*b3*tf^2 + 4*b4*tf^3 + 5*b5*tf^4   = vf
    # p''(tf) = 2*b2 + 6*b3*tf + 12*b4*tf^2 + 20*b5*tf^3           = af

    # Construct system of equations for [b3, b4, b5]
    T2 = tf*tf
    T3 = T2*tf
    T4 = T3*tf
    T5 = T4*tf

    M = np.array([
        [T3,    T4,     T5   ],
        [3*T2,  4*T3,   5*T4 ],
        [6*tf,  12*T2,  20*T3]
    ], dtype=float)

    rhs = np.array([
        pf - (b0 + b1*tf + b2*T2),         # p(tf)   - known part
        vf - (b1 + 2*b2*tf),              # p'(tf)  - known part
        af - (2*b2)                       # p''(tf) - known part
    ], dtype=float)

    # Solve for b3, b4, b5
    b3, b4, b5 = np.linalg.solve(M, rhs)

    return [b0, b1, b2, b3, b4, b5]

def evaluate_poly5(coeffs, t):
    """
    Evaluate the 5th-order polynomial at time t.
    :param coeffs : [b0, b1, b2, b3, b4, b5]
    :param : t time to evaluate at
    returns p(t) = b0 + b1 t + b2 t^2 + b3 t^3 + b4 t^4 + b5 t^5
    """
    b0, b1, b2, b3, b4, b5 = coeffs
    return (b0 + b1 * t + b2 * t**2 + b3 * t**3 + b4 * t**4 + b5 * t**5)

def evaluate_poly5_vel(coeffs, t):
    """
    Evaluate the velocity (1st derivative) of the 5th-order polynomial.
    dot p(t) = b1 + 2 b2 t + 3 b3 t^2 + 4 b4 t^3 + 5 b5 t^4
    """
    b0, b1, b2, b3, b4, b5 = coeffs
    return (b1 + 2 * b2 * t + 3 * b3 * t**2 + 4 * b4 * t**3 + 5 * b5 * t**4)

def evaluate_poly5_acc(coeffs, t):
    """
    Evaluate the acceleration (2nd derivative) of the 5th-order polynomial.
    ddot p(t) = 2 b2 + 6 b3 t + 12 b4 t^2 + 20 b5 t^3
    """
    b0, b1, b2, b3, b4, b5 = coeffs
    return (2 * b2 +  6 * b3 * t + 12 * b4 * t**2 + 20 * b5 * t**3)

def generate_trajectory_5th_roder( waypoints, velocities, accelerations, position_limits, max_velocity,  max_acceleration, tf, T):
    """
    Given a list of waypoints, a list of velocities and a list of accelerations 
    to pass over each waypoint and the total time tf to perform on each segment 
    with time interval of T is computes the successive positions and composes the trajectory.
    :param waypoints : list of waypoints (each a 3-element list (x,y,z))
    :param velocities: list of velocities (each a 3-element list (vx,vy,vz))
    :param accelerations: list of accelerations (each a 3-element list (ax, ay, azz))
    :position_limits : list of a pair of limits for each coordinate ([min_x, max_x], [min_y, max_y], [min_z, max_z])
    :max_velocity    : the maximum velocity permited in means of magnitude (norm(V))
    :max_acceleration: the maximum acceleration permited in means of magnitude (norm(a))
    :param tf        : total time for the motion
    :return          : list of waypoints and the velocities at each point if valid.
    """
    is_valid = True
    x_lim, y_lim, z_lim = position_limits

    trajectory_out   = []
    velocity_out     = []
    acceleration_out = []

    N = int(round(tf / T))  # number of points per segment

    # For each segment between waypoint k and k+1
    for k in range(len(waypoints) - 1):
        x0, y0, z0 = waypoints[k]
        x1, y1, z1 = waypoints[k + 1]

        vx0, vy0, vz0 = velocities[k]
        vx1, vy1, vz1 = velocities[k + 1]

        ax0, ay0, az0 = accelerations[k]
        ax1, ay1, az1 = accelerations[k + 1]

        # solve for polynomial coefficients in each dimension
        cx = poly5_interpolation(x0, vx0, ax0, x1, vx1, ax1, tf)
        cy = poly5_interpolation(y0, vy0, ay0, y1, vy1, ay1, tf)
        cz = poly5_interpolation(z0, vz0, az0, z1, vz1, az1, tf)

        # Local buffers
        seg_positions = []
        seg_vels      = []
        seg_accs      = []

        # Evaluate 0 <= t <= tf in increments of T
        for n in range(N):
            t = n * T

            # position
            x = evaluate_poly5(cx, t)
            y = evaluate_poly5(cy, t)
            z = evaluate_poly5(cz, t)

            # check position limits
            if not (x_lim[0] <= x <= x_lim[1]):
                print(f"[WARNING] x out of range: {x}")
                is_valid = False

            if not (y_lim[0] <= y <= y_lim[1]):
                print(f"[WARNING] y out of range: {y}")
                is_valid = False

            if not (z_lim[0] <= z <= z_lim[1]):
                print(f"[WARNING] z out of range: {z}")
                is_valid = False

            seg_positions.append([x, y, z])

            # velocity
            vx = evaluate_poly5_vel(cx, t)
            vy = evaluate_poly5_vel(cy, t)
            vz = evaluate_poly5_vel(cz, t)

            speed = np.linalg.norm([vx, vy, vz])

            if speed > max_velocity:
                print(f"[WARNING] speed exceeded: {speed} at segment {k}, t={t}")
                is_valid = False

            seg_vels.append([vx, vy, vz])

            # acceleration
            ax = evaluate_poly5_acc(cx, t)
            ay = evaluate_poly5_acc(cy, t)
            az = evaluate_poly5_acc(cz, t)

            acc_norm = np.linalg.norm([ax, ay, az])

            if acc_norm > max_acceleration:
                print(f"[WARNING] accel exceeded: {acc_norm} at segment {k}, t={t}")
                is_valid = False

            seg_accs.append([ax, ay, az])

        # remove the last point if not the final segment to prevent 
        # duplication at the next segment start
        if k < len(waypoints) - 2:
            seg_positions.pop()
            seg_vels.pop()
            seg_accs.pop()

        # Append to global arrays
        trajectory_out.extend(seg_positions)
        velocity_out.extend(seg_vels)
        acceleration_out.extend(seg_accs)

        if not is_valid:
            break

    return trajectory_out, velocity_out, acceleration_out, is_valid