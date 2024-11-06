import numpy as np
from scipy.linalg import cholesky, cho_solve, solve_triangular
import time

debug_mod = False
def cubature_int(func, mean, cov, return_cubature_points=False, cubature_points=None, u=None, sqr=False):
    """
   Computes the cubature integral of the type "\int func(x) p(x) dx", where func is an arbitrary function of x, p(x) is
   considered a Gaussian distribution with mean given by 'mean', and covariance matrix 'cov'. The integral is performed
   numerically as func_int = sum(func(sig_points(i)))/N
   The outputs are the mean vector ('func_int') and the cubature points matrix ('cubature_points').
    :param func: function to be integrated.
    :param mean: (d,) mean numpy array
    :param cov: (d,d) numpy array (covariance matrix)
    :param cubature_points: (d, 2*d) numpy array containing the 2*d cubature_points (this depends on the distribution p(x) if
                        it changes new points need to be generated.
    :param u: input signal in case of f(x,u) (default u=None, f(x) ).
    :param sqr: Boolean defining if running a square-root kf. If True than cov should be a lower triangular
    decomposition of the covariance matrix! That is cov = cholesky(P).
    :return: func_int
    -------------------------------------------------------------------
   e.g.
       cov = np.array([[1.2769, 0.0843], [0.0843, 1.0725]])
        mean = np.array([0.1, 2])
        fh = lambda x: x**2 + 10
        mean, cubature_points = cubature_int(fh, mean, cov)
        print(mean, cubature_points)
    -------------------------------------------------------------------
    Author: Tales Imbiriba
    Last modified in Jan 2021.
    """
    if debug_mod:
        start= time.time()
        
    d = len(mean)
    n_points = 2 * d

    if cubature_points is None:
        # create cubature points;
        cubature_points = gen_cubature_points(mean, cov, d, n_points, sqr)

    if debug_mod:
        print('cubature int step 1 took: ', time.time() - start, 's')
        start = time.time()
    

    int_mean = int_func(func, cubature_points, u)
    # print(int_mean)
    if debug_mod:
        print('cubature int step 2 took: ', time.time() - start, 's')
        
    if return_cubature_points:
        return int_mean, cubature_points
    else:
        return int_mean


def gen_cubature_points(mean, cov, d, n_points, sqr=False):
    cubature_points = np.zeros((n_points, d))
    
    # L is lower-triangular: cov = LxL'
    if sqr:
        L = cov
    else:
        reg = 0.000001
        L = np.linalg.cholesky(cov + reg * np.eye(len(cov)))

    num = np.sqrt(n_points / 2)
    # print(d,"d")
    # print(n_points,"n_points")
    xi = np.concatenate((num * np.eye(d), - num * np.eye(d)), axis=1).T
    for i in range(n_points):
        # print(xi.shape,"xi.shape")
        # a = mean + np.dot(L, xi[i].T).reshape(-1, 1)
        cubature_points[i] = mean + np.dot(L, xi[i])

    # cubature_points = np.dot(xi, L.T)
    # mean = np.tile(mean, (n_points,1))
    # cubature_points = cubature_points + mean
    




    # make sure the cubature_points are not being modified
    cubature_points.flags.writeable = False

    return cubature_points


def int_func(func, cubature_points, u=None):
    if u is None:
        return np.mean([func(x) for x in cubature_points], axis=0)
    else:
        return np.mean([func(x, u) for x in cubature_points], axis=0)

def inv_pd_mat(K, reg=1e-5):
    """
    Usage: inv_pd_mat(self, K, reg=1e-5)
    Invert (Squared) Positive Definite matrix using Cholesky decomposition.
    :param K: Positive definite matrix. (ndarray).
    :param reg: a regularization parameter (default: reg = 1e-6).
    :return: the inverse of K.
    """
    # compute inverse K_inv of K based on its Cholesky
    # decomposition L and its inverse L_inv
    K = K.copy() + reg * np.eye(len(K))
    L = cholesky(K, lower=True)
    L_inv = solve_triangular(L.T, np.eye(L.shape[0]))
    return L_inv.dot(L_inv.T)

# def inv_triangular_matrix(S, lower=True):
#     """
#     Invert triangular matrix
#     :param S: is a triangular matrix
#     :return: inv(S)
#     """
#     if lower:
#         S_inv = solve_triangular(S.T, np.eye(S.shape[0]))
#         return S_inv.dot(S_inv.T)
#     else:
#         S_inv = solve_triangular(S, np.eye(S.T.shape[0]))
#         return S_inv.T.dot(S_inv)


if __name__ == '__main__':
    # mean = np.array([0.1, 2])
    # cov = np.array([[1.2769, 0.0843], [0.0843, 1.0725]])

    mean = np.array([0, 0])
    # cov = np.array([[1.2769, 0.0843], [0.0843, 1.0725]])
    cov = np.array([[1, 0], [0, 1]])
    y = 7


    # fh = lambda x: x ** 2 + y
    # fh = lambda x: x
    def fh(x): return x


    # fhfht = lambda x: np.outer(fh(x), fh(x))
    def fhfht(x): return np.outer(fh(x), fh(x))


    mean, cubature_points = cubature_int(fh, mean, cov, return_cubature_points=True)
    # print(mean, cubature_points)
    # print(cubature_points)
    # print("second:")
    # print(cubature_points[0])
    mean2, cubature_points2 = cubature_int(fh, mean, cov, cubature_points=cubature_points, return_cubature_points=True)
    # mean = cubature_int(fh, mean, cov)
    # print(mean, cubature_points2)

    # print("THIRD:")
    # # cov = cubature_int(fhfht, mean, cov, cubature_points=cubature_points)
    cov2 = cubature_int(fhfht, mean, cov)
    # # print(mean, cubature_points)