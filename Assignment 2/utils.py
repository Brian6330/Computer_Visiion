import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import null_space


def get_normalization_matrix(x):
    """
    get_normalization_matrix Returns the transformation matrix used to normalize
    the inputs x
    Normalization corresponds to subtracting mean-position and positions
    have a mean distance of sqrt(2) to the center
    """
    # Input: x 3*N
    # 
    # Output: T 3x3 transformation matrix of points

    # --------------------------------------------------------------
    # Estimate transformation matrix used to normalize
    # the inputs x
    # --------------------------------------------------------------

    # Get centroid and mean-distance to centroid (source: https://cs.adelaide.edu.au/~wojtek/papers/pami-nals2.pdf)
    _T = np.zeros((3, 3))
    centroid = np.mean(x, 1)
    s = 1 / np.sqrt(
        np.sum(
            np.square(
                np.apply_along_axis(lambda v: v - centroid, 0, x)
            )
        )
        / (2 * x.shape[1])
    )

    _T[0, 0] = s
    _T[0, 2] = - s * centroid[0]
    _T[1, 1] = s
    _T[1, 2] = - s * centroid[1]
    _T[2, 2] = 1

    return _T


def eight_points_algorithm(x1, x2, normalize=True):
    """
    Calculates the fundamental matrix between two views using the normalized 8 point algorithm
    Inputs:
                    x1      3xN     homogeneous coordinates of matched points in view 1
                    x2      3xN     homogeneous coordinates of matched points in view 2
    Outputs:
                    F       3x3     fundamental matrix
    """
    _N = x1.shape[1]

    if normalize:
        # Construct transformation matrices to normalize the coordinates
        _Tx1 = get_normalization_matrix(x1)
        _Tx2 = get_normalization_matrix(x2)

        # Normalize inputs
        n1 = np.apply_along_axis(lambda x: np.matmul(_Tx1, x), 0, x1)
        n2 = np.apply_along_axis(lambda x: np.matmul(_Tx2, x), 0, x2)

    # Construct matrix A encoding the constraints on x1 and x2
    _A = np.zeros((_N, 9))
    for i in range(_N):
        _A[i, 0] = n2[0, i] * n1[0, i]
        _A[i, 1] = n2[0, i] * n1[1, i]
        _A[i, 2] = n2[0, i]
        _A[i, 3] = n2[1, i] * n1[0, i]
        _A[i, 4] = n2[1, i] * n1[1, i]
        _A[i, 5] = n2[1, i]
        _A[i, 6] = n1[0, i]
        _A[i, 7] = n1[1, i]
        _A[i, 8] = 1

    # Solve for _F using SVD
    u, s, v = np.linalg.svd(_A)
    _F = v.T[:, 8].reshape(3, 3)

    # Enforce that rank(_F)=2
    u, s, v = np.linalg.svd(_F)

    # Remove smallest singular value as described on slide 33 in lecture 08a_epipolar
    s_temp = np.array([[s[0], 0, 0],
                       [0, s[1], 0],
                       [0, 0, 0]])
    _F = np.matmul(np.matmul(u, s_temp), v)

    # Transform F back
    if normalize:
        _F = np.matmul(np.matmul(_Tx2.T, _F), _Tx1)

    return _F


def right_epipole(_F):
    """
    Computes the (right) epipole from a fundamental matrix F.
    (Use with F.T for left epipole.)
    Description of epipole in Jupyter Notebook.
    """

    # The epipole is the null space of F (F * e = 0)
    e = null_space(_F)
    return e / e[2]


def plot_epipolar_line(im, _F, x, e, plot=plt):
    """
    Plot the epipole and epipolar line F*x=0 in an image. F is the fundamental matrix
    and x a point in the other image.
    Description of epipolar lines in Jupyter Notebook
    """
    m, n = im.shape[:2]
    epipolar_line = np.dot(_F, x)

    x = np.linspace(0, n, 4)
    y = np.array([(epipolar_line[2] + epipolar_line[0] * x_coord) / (-epipolar_line[1]) for x_coord in x])

    plot.plot(x[(y >= 0) & (y < m)], y[(y >= 0) & (y < m)], linewidth=2)
