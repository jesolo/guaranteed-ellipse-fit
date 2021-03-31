# Author: Jedrzej Solarski



import math
import statistics as stat
import sys
from ctypes import *
from dataclasses import dataclass


import numpy as np
import scipy.linalg


__all__ = [
    "ellipse_matrix",
    "fit_ellipse",
    "ellipse_to_points",
    "unit_vector",
    "angle_between",
    "angle_between2",
    "point_ring",
    "ellipse_matrix_to_ax",
    "normalize_data_isotropically",
    "compute_directellipse_estimates",
    "fastLevenbergMarquardtStep",
    "fastGuaranteedEllipseFit",
    "fast_guaranteed_ellipse_estimate",
]



@dataclass
class MyStructure:
    def __init__(self, dataPts, normalised_CovList, maxIter=200):
        self.maxIter = maxIter
        self.eta_updated = False
        self.lambdaa = 0.01
        self.k = 0
        self.damping_multiplier = 15
        self.damping_divisor = 1.2
        self.numberOfPoints = len(dataPts)
        self.data_points = dataPts
        self.covList = normalised_CovList
        self.tolDelta = 1e-7
        self.tolCost = 1e-7
        self.tolEta = 1e-7
        self.tolGrad = 1e-7
        self.tolBar = 15.5
        self.tolDet = 1e-5
        self.cost = np.zeros((1, maxIter + 1))
        self.eta = np.zeros((5, maxIter + 1))
        self.t = np.zeros((6, maxIter + 1))
        self.delta = np.zeros((5, maxIter + 1))
        # self.r = np.zeros((self.numberOfPoints,1))
        # self.jacobian_matrix = np.zeros((self.numberOfPoints,5))
        self.H = []
        self.jacob_latentParameters = []


def ellipse_matrix(C):
    return np.array(
        [
            [C[0], C[1] / 2.0, C[3] / 2],
            [C[1] / 2, C[2], C[4] / 2],
            [C[3] / 2, C[4] / 2, C[5]],
        ]
    )


def fit_ellipse(ellipse_points):
    x, y = ellipse_points[:, 0], ellipse_points[:, 1]
    A = np.stack([x ** 2, x * y, y ** 2, x, y, np.ones_like(x)], axis=1)
    U, S, V = np.linalg.svd(A)
    return V[-1], S[-1]





def ellipse_to_points(Ct, num_points, angle=360.0, vstart=(1, 0)):
    """Returns points of ellipse form ellipse matrix

    Parameters
    ----------
    Ct: np.ndarray
        ellipse matrix
    num_points: int
        desired number of poits
    angle: float
        angle that spans ellipse starts from vstart point
    vstart: (float,float) or [float,float]
        starting vector for ellipse spanning a tuple or a list of coordinates (x,y)

    Returns
    -------
    points: np.ndarray
        list of points for ellipse

    """
    A = Ct[0]
    B = Ct[1]
    C = Ct[2]
    D = Ct[3]
    E = Ct[4]
    F = Ct[5]
    denomin = B ** 2 - 4 * A * C

    x0 = (2 * C * D - B * E) / denomin
    y0 = (2 * A * E - B * D) / denomin
    t0 = [x0, y0]
    a = (
        -(
            (
                2
                * (A * E ** 2 + C * D ** 2 - B * D * E + (B ** 2 - 4 * A * C) * F)
                * ((A + C) + ((A - C) ** 2 + B ** 2) ** 0.5)
            )
            ** 0.5
        )
        / denomin
    )

    b = (
        -(
            (
                2
                * (A * E ** 2 + C * D ** 2 - B * D * E + (B ** 2 - 4 * A * C) * F)
                * ((A + C) - ((A - C) ** 2 + B ** 2) ** 0.5)
            )
            ** 0.5
        )
        / denomin
    )
    if math.isnan(b):
        b = 0
    if B == 0 and A > C:
        beta = np.pi / 2
    elif B == 0 and A <= C:
        beta = 0
    else:
        beta = math.atan((C - A - ((A - C) ** 2 + B ** 2) ** 0.5) / B)
    if math.isnan(beta):
        beta = 0

    sinbeta = math.sin(beta)
    cosbeta = math.cos(beta)

    points = []
    step = angle / num_points
    for k in range(0, num_points):
        # generate points by rotating vector by angle (step) according to ellipse
        # equation starting from vector vstart (default [1,0])
        i = step * k + angle_between([1, 0], vstart) * 180 / np.pi
        alpha = i * (math.pi / 180)
        sinalpha = math.sin(alpha)
        cosalpha = math.cos(alpha)

        X = t0[0] + (a * cosalpha * cosbeta - b * sinalpha * sinbeta)
        Y = t0[1] + (a * cosalpha * sinbeta + b * sinalpha * cosbeta)
        points.append([X, Y])
    return np.array(points)


def rotate(x, y, rad):
    """Two dimensional rotation

    Parameters
    ----------
    x,y: float
        x and y point coordinate
    r: float
        angle in radians

    Returns
    -------
    (float,float)
        coordinates of rotated point in a tuple (x,y)
    """
    rx = (x * math.cos(rad)) - (y * math.sin(rad))
    ry = (y * math.cos(rad)) + (x * math.sin(rad))
    return (rx, ry)


def unit_vector(vector):
    """ Returns the unit vector of the vector."""
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors v1 and v2"""

    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def angle_between2(v1, v2):
    """ Returns the angle in radians in range [0, 2PI] between vectors v1 and v2 clockwise"""
    v1 = unit_vector(v1)
    v2 = unit_vector(v2)
    xself = v1[0]
    yself = v1[1]
    xother = v2[0]
    yother = v2[1]
    ret = math.atan2(yother, xother) - math.atan2(yself, xself)
    if ret < 0:
        ret = ret + 2 * math.pi
    return ret


def point_ring(center, num_points, radius):
    """Create a ring of points

    Ring of points centered on point (x,y) with a given radius
    using the specified number of points
    returns a list of point coordinates in tuples

    Parameters
    ----------
    center: (float,float) or [float,float]
        center o the circle
    num_points: int
        desired number of points
    radius: float
        desired radius

    Returns
    -------
    points: np.ndarray
        list of points for created circle
    """
    arc = (2 * math.pi) / num_points  # what is the angle between two of the points
    points = []
    for p in range(num_points):
        (px, py) = rotate(0, radius, arc * p)
        px += center[0]
        py += center[1]
        points.append([px, py])
    return np.array(points)


def ellipse_matrix_to_ax(Ct):
    """Returns height, width, rotation angle and centero f ellipse form ellipse matrix

    Parameters
    ----------
    Ct: np.ndarray
        ellipse matrix

    Returns
    -------
    points: floats
        both semi-axis and center point of ellipse

    """

    A = Ct[0]
    B = Ct[1]
    C = Ct[2]
    D = Ct[3]
    E = Ct[4]
    F = Ct[5]
    denomin = B ** 2 - 4 * A * C
    x0 = (2 * C * D - B * E) / denomin
    y0 = (2 * A * E - B * D) / denomin
    t0 = [x0, y0]
    a = (
        -(
            (
                2
                * (A * E ** 2 + C * D ** 2 - B * D * E + (B ** 2 - 4 * A * C) * F)
                * ((A + C) + ((A - C) ** 2 + B ** 2) ** 0.5)
            )
            ** 0.5
        )
        / denomin
    )
    b = (
        -(
            (
                2
                * (A * E ** 2 + C * D ** 2 - B * D * E + (B ** 2 - 4 * A * C) * F)
                * ((A + C) - ((A - C) ** 2 + B ** 2) ** 0.5)
            )
            ** 0.5
        )
        / denomin
    )

    if math.isnan(b):
        b = 0
    if B == 0 and A > C:
        beta = np.pi / 2
    elif B == 0 and A <= C:
        beta = 0
    else:
        beta = math.atan((C - A - ((A - C) ** 2 + B ** 2) ** 0.5) / B)

    if math.isnan(beta):
        beta = 0

    return beta, a, b, t0


def normalize_data_isotropically(data_points):
    '''

    Description: This procedure takes as input a matrix of two-dimensional
              coordinates and normalizes the coordinates so that they
              lie inside a unit box.

    Parameters : dataPts               - an nPoints x 2 matrix of
                                      coordinates

    Return     : normalizedPts         - an nPoints x 2 matrix of
                                      coordinates which are constrained
                                      to lie inside a unit box.

                : T                     - a 3x3 affine transformation matrix
                                      T that was used to transform the
                                      (homogenous coordinates) of the data
                                      points so that they lie inside a
                                      unit box.


    '''
    
    nPoints = len(data_points)
    new_col = np.ones((nPoints, 1))
    new_data = np.append(data_points, new_col, axis=1)
    meanX = stat.mean(new_data[:, 0])
    meanY = stat.mean(new_data[:, 1])
    s = (
        (1 / (2 * nPoints))
        * sum((new_data[:, 0] - meanX) ** 2 + (new_data[:, 1] - meanY) ** 2)
    ) ** 0.5
    T = [
        [s ** (-1), 0, -(s ** (-1)) * meanX],
        [0, s ** (-1), -(s ** (-1)) * meanY],
        [0, 0, 1],
    ]
    normalizedPts = T @ new_data.T
    normalizedPts = normalizedPts.T
    normalizedPts = np.delete(normalizedPts, 2, 1)
    return normalizedPts, T


def compute_directellipse_estimates(data_points):
    '''

    This function is a wrapper for the numerically stable direct ellipse
    fit due to

    R. Halif and J. Flusser
    "Numerically stable direct least squares fitting of ellipses"
    Proc. 6th International Conference in Central Europe on Computer
    Graphics and Visualization. WSCG '98 Czech Republic,125--132, feb, 1998
    Parameters:

       dataPts    - a Nx2 matrix where N is the number of data points

    Returns:

      a length-6 vector [a b c d e f] representing the parameters of the
      equation
      a x^2 + b x y + c y^2 + d x + e y + f = 0
      with the additional result that b^2 - 4 a c < 0.
    '''
    nPts = len(data_points)
    new_col = np.ones((nPts, 1))
    normalizedPoints, T = normalize_data_isotropically(data_points)
    normalizedPoints = np.append(normalizedPoints, new_col, axis=1)
    theta, _ = fit_ellipse(normalizedPoints)
    C = ellipse_matrix(theta)
    T = np.array(T)
    X = T.T @ C @ T
    C = [
        X[0, 0],
        2 * X[0, 1],
        X[1, 1],
        2 * X[0, 2],
        2 * X[1, 2],
        X[2, 2],
    ]
    C = C / np.linalg.norm(C)
    return C


def fastLevenbergMarquardtStep(struct, rho=2):
    '''
       This function is used in the main loop of guaranteedEllipseFit in the
    process of minimizing an approximate maximum likelihood cost function
    This function is used in the main loop of guaranteedEllipseFit in the
    process of minimizing an approximate maximum likelihood cost function
    of an ellipse fit to data.  It computes an update for the parameters
    representing the ellipse, using the method of Levenberg-Marquardt for
    non-linear optimisation.
    See: http://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm

    However, unlike the traditional LevenbergMarquardt step, we do not
    add a multiple of the identity matrix to the approximate Hessian,
    but instead a different positive semi-definite matrix. Our choice
    particular choice of the different matrix corresponds to the
    gradient descent direction in the theta coordinate system,
    transformed to the eta coordinate system. We found empirically
    that taking steps according to the theta coordinate system
    instead of the eta coordinate system lead to faster convergence.


    Parameters:
       struct     - a data structure containing various parameters
                    needed for the optimisation process.

    Returns:

      the same data structure 'struct', except that relevant fields have
      been updated%   representing the ellipse, using the method of Levenberg-Marquardt for
     of an ellipse fit to data.  It computes an update for the parameters
    non-linear optimisation.
    See: http://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm

    However, unlike the traditional LevenbergMarquardt step, we do not
    add a multiple of the identity matrix to the approximate Hessian,
    but instead a different positive semi-definite matrix. Our choice
    particular choice of the different matrix corresponds to the
    gradient descent direction in the theta coordinate system,
    transformed to the eta coordinate system. We found empirically
    that taking steps according to the theta coordinate system
    instead of the eta coordinate system lead to faster convergence.

    Parameters:

       struct     - a data structure containing various parameters
                    needed for the optimisation process.

    Returns:

      the same data structure 'struct', except that relevant fields have
      been updated
    '''
    jacobian_matrix = struct.jacobian_matrix
    r = struct.r
    lambada = struct.lambdaa
    delta = struct.delta[0][struct.k]
    damping_multiplier = struct.damping_multiplier
    damping_divisor = struct.damping_divisor
    current_cost = struct.cost[0][struct.k]
    data_points = struct.data_points
    covList = struct.covList
    numberOfPoints = struct.numberOfPoints
    H = struct.H
    jlp = np.array(struct.jacob_latentParameters)
    eta = np.array(struct.eta[:, struct.k])

    t = np.array(
        [1, 2 * eta[0], eta[0] ** 2 + abs(eta[1]) ** 2, eta[2], eta[3], eta[4]],
        dtype=float,
    )
    t = t / np.linalg.norm(t)
    jacob = jacobian_matrix.T @ r
    DMP = (jlp.T @ jlp) * lambada
    update_a = np.linalg.lstsq(-(H + DMP), jacob, rcond=None)[0]
    DMP = (jlp.T @ jlp) * lambada / damping_divisor
    update_b = np.linalg.lstsq(-(H + DMP), jacob, rcond=None)[0]
    eta_potential_a = (eta + update_a.T).T
    eta_potential_b = (eta + update_b.T).T
    t_potential_a = np.array(
        [
            1,
            2 * eta_potential_a[0] ** 2,
            eta_potential_a[0] ** 2 + abs(eta_potential_a[1]) ** 2,
            eta_potential_a[2],
            eta_potential_a[3],
            eta_potential_a[4],
        ],
        dtype=float,
    )
    t_potential_a = (t_potential_a / np.linalg.norm(t_potential_a)).T
    t_potential_b = np.array(
        [
            1,
            2 * eta_potential_b[0] ** 2,
            eta_potential_a[0] ** 2 + abs(eta_potential_b[1]) ** 2,
            eta_potential_b[2],
            eta_potential_b[3],
            eta_potential_b[4],
        ],
        dtype=float,
    )
    t_potential_b = (t_potential_b / np.linalg.norm(t_potential_b)).T
    cost_a = 0
    cost_b = 0
    for i in range(struct.numberOfPoints):
        m = np.array(data_points)
        m = m[i, :]
        ux_i = np.array([m[0] ** 2, m[0] * m[1], m[1] ** 2, m[0], m[1], 1]).T
        dux_i = np.array([[2 * m[0], m[1], 0, 1, 0, 0], [0, m[0], 2 * m[1], 0, 1, 0]]).T

        A = np.outer(ux_i, ux_i.T)

        covX_i = covList[i]

        B = dux_i @ covX_i @ dux_i.T

        t_aBt_a = t_potential_a.T @ B @ t_potential_a
        t_aAt_a = t_potential_a.T @ A @ t_potential_a

        t_bBt_b = t_potential_b.T @ B @ t_potential_b
        t_bAt_b = t_potential_b.T @ A @ t_potential_b

        cost_a = cost_a + abs(t_aAt_a / t_aBt_a)
        cost_b = cost_b + abs(t_bAt_b / t_bBt_b)

    if cost_a >= current_cost and cost_b >= current_cost:
        struct.eta_updated = False

        struct.cost[0][struct.k + 1] = current_cost

        struct.eta[:, struct.k + 1] = eta

        struct.t[:, struct.k + 1] = t

        struct.delta[:, struct.k + 1] = delta

        struct.lambdaa = lambada * damping_multiplier
    elif cost_b < current_cost:

        struct.eta_updated = True

        struct.cost[0][struct.k + 1] = cost_b

        struct.eta[:, struct.k + 1] = eta_potential_b.T

        struct.t[:, struct.k + 1] = t_potential_b

        struct.delta[:, struct.k + 1] = update_b.T

        struct.lambdaa = lambada / damping_divisor
    else:

        struct.eta_updated = True

        struct.cost[0][struct.k + 1] = cost_a

        struct.eta[:, struct.k + 1] = eta_potential_a.T

        struct.t[:, struct.k + 1] = t_potential_a

        struct.delta[:, struct.k + 1] = update_a.T

        struct.lambdaa = lambada
    return struct


def fastGuaranteedEllipseFit(latentParameters, data_points, covList):
    '''
    This function implements the ellipse fitting algorithm described in
    Z.Szpak, W. Chojnacki and A. van den Hengel
    "Guaranteed Ellipse Fitting with an Uncertainty Measure for Centre,
     Axes, and Orientation"


    Parameters:

       latentParameters    - an initial seed for latent parameters
                             [p q r s t] which through a transformation
                             are related to parameters  [a b c d e f]
                             associated with the conic equation

                              a x^2 + b x y + c y^2 + d x + e y + f = 0

       dataPts             - a 2xN matrix where N is the number of data
                             points

       covList             - a list of N 2x2 covariance matrices
                             representing the uncertainty of the
                             coordinates of each data point.


    Returns:

      a length-6 vector [a b c d e f] representing the parameters of the
      equation

      a x^2 + b x y + c y^2 + d x + e y + f = 0

      with the additional result that b^2 - 4 a c < 0.

    '''
    eta = np.array(latentParameters)
    t = np.array(
        (1, 2 * eta[0], eta[0] ** 2 + abs(eta[1]) ** 2, eta[2], eta[3], eta[4])
    ).T
    t = t / np.linalg.norm(t)
    struct = MyStructure(data_points, covList)
    Fprim = np.array(([0, 0, 2], [0, -1, 0], [2, 0, 0]))
    F = np.zeros((6, 6))
    F[0:3, 0:3] = Fprim
    I = np.identity(6)
    maxIter = struct.maxIter
    keep_going = True
    struct.eta[:, struct.k] = eta
    struct.t[:, struct.k] = t
    struct.delta[:, struct.k] = np.ones((1, 5))
    while keep_going and struct.k < maxIter:
        struct.r = np.zeros((struct.numberOfPoints, 1))
        struct.jacobian_matrix = np.zeros((struct.numberOfPoints, 5))
        eta = struct.eta[:, struct.k]  # eta = struct.eta[struct.k]

        t = np.array(
            (1, 2 * eta[0], eta[0] ** 2 + abs(eta[1]) ** 2, eta[2], eta[3], eta[4])
        ).T
        jacob_latentParameters = [
            [0, 0, 0, 0, 0],
            [2, 0, 0, 0, 0],
            [2 * eta[0], 2 * abs(eta[1]) ** (2 - 1) * np.sign(eta[1]), 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
        ]
        Pt = np.identity(6) - (np.outer(t.T, t) / (np.linalg.norm(t, 2) ** 2))
        jacob_latentParameters = (
            (1 / np.linalg.norm(t, 2)) * Pt @ jacob_latentParameters
        )
        t = t / np.linalg.norm(t)
        for i in range(struct.numberOfPoints):
            m = np.array(data_points)
            m = m[i, :]
            ux_i = np.array([m[0] ** 2, m[0] * m[1], m[1] ** 2, m[0], m[1], 1]).T
            dux_i = np.array(
                [[2 * m[0], m[1], 0, 1, 0, 0], [0, m[0], 2 * m[1], 0, 1, 0]]
            ).T

            A = np.outer(ux_i, ux_i.T)

            covX_i = covList[i]

            B = dux_i @ covX_i @ dux_i.T

            tBt = t.T @ B @ t
            tAt = t.T @ A @ t

            struct.r[i] = (abs(tAt / tBt)) ** 0.5

            M = A / tBt
            Xbits = B * ((tAt) / (tBt ** 2))
            X = M - Xbits

            grad = (X @ t.T) / ((abs(tAt / tBt) + np.e)) ** 0.5

            struct.jacobian_matrix[i, :] = grad @ jacob_latentParameters
        struct.H = struct.jacobian_matrix.T @ struct.jacobian_matrix
        struct.cost[0][struct.k] = np.dot(struct.r.T, struct.r)
        struct.jacob_latentParameters = jacob_latentParameters
        struct = fastLevenbergMarquardtStep(struct)
        eta = struct.eta[:, struct.k + 1]
        t = np.array(
            (1, 2 * eta[0], eta[0] ** 2 + abs(eta[1]) ** 2, eta[2], eta[3], eta[4])
        ).T
        t = t / np.linalg.norm(t)
        tIt = t.T @ I @ t
        tFt = t.T @ F @ t
        barrier = tIt / tFt
        M = [
            [t[0], t[1] / 2, t[3] / 2],
            [t[1] / 2, t[2], t[4] / 2],
            [t[3] / 2, t[4] / 2, t[5]],
        ]
        DeterminantConic = np.linalg.det(M)
        if (
            min(
                np.linalg.norm(struct.eta[:, struct.k + 1] - struct.eta[:, struct.k]),
                np.linalg.norm(struct.eta[:, struct.k + 1] + struct.eta[:, struct.k]),
            )
            < struct.tolEta
            and struct.eta_updated
        ):
            keep_going = False
        elif (
            abs(struct.cost[0][struct.k] - struct.cost[0][struct.k + 1])
            < struct.tolCost
            and struct.eta_updated
        ):
            keep_going = False
        elif (
            np.linalg.norm(struct.delta[:, struct.k + 1]) < struct.tolDelta
            and struct.eta_updated
        ):
            keep_going = False
        elif np.linalg.norm(grad) < struct.tolGrad:
            keep_going = False
        elif np.log(barrier) > struct.tolBar or abs(DeterminantConic) < struct.tolDet:
            keep_going = False
        struct.k = struct.k + 1
    iterations = struct.k
    theta = struct.t[:, struct.k]
    theta = theta / np.linalg.norm(theta)
    return theta, iterations


def fast_guaranteed_ellipse_estimate(data_points, covList=None):
    '''
    Description: This procedure takes as input a matrix of two-dimensional
             coordinates and estimates a best fit ellipse using the
             sampson distance between a data point and the ellipse
             equations as the error measure. The Sampson distance is
             an excellent approximation to the orthogonal distance for
             small noise levels. The Sampson distance is often also
             referred to as the approximate maximum likelihood (AML).
             The user can specify a list of covariance matrices for the
             data points. If the user does not specify a list of
             covariance matrices then isotropic homogeneous Gaussian
             noise is assumed.
    Parameters : initialParameters    - initial parameters use to seed the
                                    iterative estimation process
             dataPts                - an nPoints x 2 matrix of
                                     coordinates
             covList               - a list of N 2x2 covariance matrices
                                     representing the uncertainty of the
                                     coordinates of each data point.
                                     if this parameter is not specified
                                     then  default isotropic  (diagonal)
                                     and homogeneous (same noise level
                                     for each data point) covariance
                                     matrices are assumed.
    Return     : a length-6 vector containing an estimate of the ellipse
             parameters theta = [a b c d e f] associated with the ellipse
             equation
                  a*x^2+ b * x y + c * y^2 + d * x + e*y + f = 0

             with the additional result that b^2 - 4 a c < 0.
    '''
    nPts = len(data_points)
    E = np.diag([1, 2 ** -1, 1, 2 ** -1, 2 ** -1, 1])
    data_pts, T = normalize_data_isotropically(data_points)
    T = np.array(T)
    if covList == None:
        covList = [np.identity(2) for i in range(nPts)]
        normalised_CovList = []
        for i in range(nPts):
            covX_i = np.zeros((3, 3))
            covX_i[0:2, 0:2] = covList[i]
            covX_i = T @ covX_i @ T.T
            normalised_CovList.append(covX_i[0:2, 0:2])

    else:
        normalised_CovList = covList
    P34 = np.kron(np.diag([0, 1, 0]), [[0, 1], [1, 0]]) + np.kron(
        np.diag([1, 0, 1]), [[1, 0], [0, 1]]
    )
    D3 = [
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
    ]
    initialEllipseParameters = compute_directellipse_estimates(data_points)
    denom = (
        P34
        @ np.linalg.pinv(D3)
        @ np.linalg.inv(np.kron(T, T)).T
        @ D3
        @ P34
        @ E
        @ initialEllipseParameters
    )
    initialEllipseParametersNormalizedSpace = np.linalg.lstsq(E, denom, rcond=None)[0]
    para = initialEllipseParametersNormalizedSpace
    p = para[1] / (2 * para[0])
    q = (para[2] / para[0] - (para[1] / (2 * para[0]) ** 2)) ** (1 / 2)
    r = para[3] / para[0]
    s = para[4] / para[0]
    t = para[5] / para[0]

    latentParameters = [p, q, r, s, t]

    ellipseParametersFinal, iterations = fastGuaranteedEllipseFit(
        latentParameters, data_pts, normalised_CovList
    )
    ellipseParametersFinal = ellipseParametersFinal / np.linalg.norm(
        ellipseParametersFinal
    )

    denom = (
        P34
        @ np.linalg.pinv(D3)
        @ (np.kron(T, T)).T
        @ D3
        @ P34
        @ E
        @ ellipseParametersFinal
    )

    estimatedParameters = np.linalg.lstsq(E, denom, rcond=None)[0]

    estimatedParameters = estimatedParameters / np.linalg.norm(estimatedParameters)

    estimatedParameters = estimatedParameters / np.sign(estimatedParameters[-1])

    return estimatedParameters, iterations
