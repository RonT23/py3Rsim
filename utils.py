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
