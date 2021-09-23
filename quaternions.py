import numpy as np


def euler2quat(angle):
    """Requires angle in order of Z, Y, X
    """
    yaw = angle[:, 0]
    pitch = angle[:, 1]
    roll = angle[:, 2]
    cy = np.cos(yaw * 0.5);
    sy = np.sin(yaw * 0.5);
    cp = np.cos(pitch * 0.5);
    sp = np.sin(pitch * 0.5);
    cr = np.cos(roll * 0.5);
    sr = np.sin(roll * 0.5);

    w = cr * cp * cy + sr * sp * sy;
    x = sr * cp * cy - cr * sp * sy;
    y = cr * sp * cy + sr * cp * sy;
    z = cr * cp * sy - sr * sp * cy;

    return np.array([w, x, y, z]).T


def quat2euler(quat):
    """Requires quaternion in form of [w, x, y, z]"""
    w, x, y, z = quat
    test = x * y + z * w
    if test > 0.499:  # singularity at north pole
        yaw = 2 * np.arctan2(x, w)
        pitch = np.pi / 2
        roll = 0
        return np.array([roll, yaw, pitch]) * 180 / np.pi

    if test < -0.499:  # singularity at south pole
        yaw = -2 * np.arctan2(x, w)
        pitch = -np.pi / 2
        roll = 0
        return np.array([roll, yaw, pitch]) * 180 / np.pi

    sqx = x * x
    sqy = y * y
    sqz = z * z
    yaw = np.arctan2(2 * y * w - 2 * x * z, 1 - 2 * sqy - 2 * sqz)
    pitch = np.arcsin(2 * test)
    roll = np.arctan2(2 * x * w - 2 * y * z, 1 - 2 * sqx - 2 * sqz)

    return np.array([roll, yaw, pitch]) * 180 / np.pi


def quat2mat(quat):
    qw = quat[:, 0]
    qx = quat[:, 1]
    qy = quat[:, 2]
    qz = quat[:, 3]

    sqrx = np.square(qx)
    sqry = np.square(qy)
    sqrz = np.square(qz)

    rot = np.zeros((quat.shape[0], 3, 3))
    rot[:, 0, 0] = 1.0 - 2.0 * (sqry + sqrz)
    rot[:, 0, 1] = 2.0 * (qx * qy - qz * qw)
    rot[:, 0, 2] = 2.0 * (qx * qz + qy * qw)

    rot[:, 1, 0] = 2.0 * (qx * qy + qz * qw)
    rot[:, 1, 1] = 1.0 - 2.0 * (sqrx + sqrz)
    rot[:, 1, 2] = 2.0 * (qy * qz - qx * qw)

    rot[:, 2, 0] = 2.0 * (qx * qz - qy * qw)
    rot[:, 2, 1] = 2.0 * (qy * qz + qx * qw)
    rot[:, 2, 2] = 1.0 - 2.0 * (sqrx + sqry)

    return rot

def angleaxis(quat):
    qw = quat[:, 0]
    qx = quat[:, 1]
    qy = quat[:, 2]
    qz = quat[:, 3]
    mag = np.sqrt(qw ** 2 + qx ** 2 + qy ** 2 + qz ** 2)
    angle = 2 * np.arccos(qw / mag)
    sin_factor = np.sqrt(1.0 - (qw / mag) ** 2)
    sin_factor[sin_factor == 0] = 1e-4
    axis = np.array([qx, qy, qz]) / sin_factor
    return angle.T, axis.T



if __name__ == "__main__":
    a1 = np.array([[90.0, 0, 0], [0, 90.0, 0], [0, 0, 90.0], [45.0, 90.0, 0]])
    a1 *= np.pi / 180.0
    quat = euler2quat(a1)
    print(quat)
    rot = quat2mat(quat)

    print(rot)

    eul = quat2euler(np.array([0.7071068, 0, 0.7071068, 0.0]))
    print(eul)

    angle_axis = angleaxis(quat)
    print("Angle axis:\n")
    print(angle_axis)
