import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import lsmr


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

    # Get centroid and mean-distance to centroid

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
        # Transformation matrices to normalize coordinates
        _Tx1 = get_normalization_matrix(x1)
        _Tx2 = get_normalization_matrix(x2)

        # Normalize inputs
        n1 = np.apply_along_axis(lambda x: np.matmul(_Tx1, x), 0, x1)
        n2 = np.apply_along_axis(lambda x: np.matmul(_Tx2, x), 0, x2)

    else:
        n1, n2 = x1, x2

    # Encode constraints on x1 and x2
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


def ransac(x1, x2, threshold, num_steps=1000, random_seed=42):
    if random_seed is not None:
        np.random.seed(random_seed)  # Random seed ensures reproducibility

    # Setup Variables
    optimal_inliers = None
    optimal_inliers_count = 0

    for _ in range(num_steps):
        # Sample 8 point correspondences randomly
        chosen_points = np.random.choice(x1.shape[1], 8, replace=False)
        temporary_fundamental_mat = eight_points_algorithm(x1[:, chosen_points], x2[:, chosen_points])

        error_signifier = np.square(np.sum(x2 * np.matmul(temporary_fundamental_mat, x1), axis=0))
        current_inliers = error_signifier < threshold
        current_inliers_count = current_inliers.sum()

        # _F is current optimum if # of inliers > # of previous highest inlier count
        if current_inliers_count > optimal_inliers_count:
            optimal_inliers_count = current_inliers_count
            optimal_inliers = current_inliers

    while True:
        _F = eight_points_algorithm(x1[:, optimal_inliers], x2[:, optimal_inliers])
        error_signifier = np.square(np.sum(x2 * (np.matmul(_F, x1)), axis=0))
        inliers = error_signifier < threshold

        if (inliers == optimal_inliers).all():
            return _F, inliers  # _F: Estimated fundamental matrix; inliers: indicator numpy array (bool)

        optimal_inliers = inliers


def decompose_essential_matrix(_E, x1, x2):
    """
    Decomposes E into a rotation and translation matrix using the
    normalized corresponding points x1 and x2.
    """

    # Fix left camera-matrix
    _Rl = np.eye(3)
    tl = np.array([[0, 0, 0]]).T
    _Pl = np.concatenate((_Rl, tl), axis=1)

    # Compute possible rotations and translations
    # Source: https://stackoverflow.com/questions/22807039/decomposition-of-essential-matrix-validation-of-the-four-possible-solutions-for
    _U, _S, _V = np.linalg.svd(_E)
    if np.linalg.det(_U) < 0:
        _U = -_U
    if np.linalg.det(_V) < 0:
        _V = -_V

    _W = np.array([[0, -1, 0],
                   [1, 0, 0],
                   [0, 0, 1]])

    _R1 = np.matmul(np.matmul(_U, _W.T), _V)
    _R2 = np.matmul(np.matmul(_U, _W), _V)

    t1 = _U[:, 2].reshape(3, 1)
    t2 = -t1

    # 4 different possibilities
    _Pr = [np.concatenate((_R1, t1), axis=1), np.concatenate((_R1, t2), axis=1),
           np.concatenate((_R2, t1), axis=1), np.concatenate((_R2, t2), axis=1)]

    # Reconstructions for all possible _R camera-matrices
    X3Ds = [infer_3d(x1[:, 0:1], x2[:, 0:1], _Pl, x) for x in _Pr]

    # Projections on image-planes
    test = [np.prod(np.hstack((_Pl @ np.vstack((X3Ds[i], [[1]])), _Pr[i] @ np.vstack((X3Ds[i], [[1]])))) > 0, 1)
            for i
            in range(4)]
    test = np.array(test)

    # Where both cameras see the point
    idx = np.where(np.hstack((test[0, 2], test[1, 2], test[2, 2], test[3, 2])) > 0.)[0][0]

    # Select correct matrix
    _Pr = _Pr[idx]

    return _Pl, _Pr


def infer_3d(x1, x2, _Pl, _Pr):
    # INFER3D Infers 3d-positions of the point-correspondences x1 and x2, using
    # the rotation matrices Rl, Rr and translation vectors tl, tr. Using a
    # least-squares approach.

    _M = x1.shape[1]

    # _R: Rotation; t: translation
    _Rl = _Pl[:3, :3]
    tl = _Pl[:3, 3]
    _Rr = _Pr[:3, :3]
    tr = _Pr[:3, 3]

    # Constraints on 3d points
    row_idx = np.tile(np.arange(4 * _M), (3, 1)).T.reshape(-1)
    col_idx = np.tile(np.arange(3 * _M), (1, 4)).reshape(-1)

    _A = np.zeros((4 * _M, 3))
    _A[:_M, :3] = x1[0:1, :].T @ _Rl[2:3, :] - np.tile(_Rl[0:1, :], (_M, 1))
    _A[_M:2 * _M, :3] = x1[1:2, :].T @ _Rl[2:3, :] - np.tile(_Rl[1:2, :], (_M, 1))
    _A[2 * _M:3 * _M, :3] = x2[0:1, :].T @ _Rr[2:3, :] - np.tile(_Rr[0:1, :], (_M, 1))
    _A[3 * _M:4 * _M, :3] = x2[1:2, :].T @ _Rr[2:3, :] - np.tile(_Rr[1:2, :], (_M, 1))

    _A = sparse.csr_matrix((_A.reshape(-1), (row_idx, col_idx)), shape=(4 * _M, 3 * _M))

    # Construct vector b
    b = np.zeros((4 * _M, 1))
    b[:_M] = np.tile(tl[0], (_M, 1)) - x1[0:1, :].T * tl[2]
    b[_M:2 * _M] = np.tile(tl[1], (_M, 1)) - x1[1:2, :].T * tl[2]
    b[2 * _M:3 * _M] = np.tile(tr[0], (_M, 1)) - x2[0:1, :].T * tr[2]
    b[3 * _M:4 * _M] = np.tile(tr[1], (_M, 1)) - x2[1:2, :].T * tr[2]

    # Solve for 3d-positions via least-squares
    w = lsmr(_A, b)[0]
    x3d = w.reshape(_M, 3).T

    return x3d
