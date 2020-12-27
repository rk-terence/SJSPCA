"""
APGD: Accelerated Proximal Gradient Descent.
Author: rkterence@zju.edu.cn
Written with python 3.7.2
"""

from numpy.linalg import norm  # the default input is 2-norm or frobenius norm.
import numpy as np
import warnings


def mixed_norm(X, p, q):
    """
    Calculate the L(p, q) norm of X.
    The input should be 2darray.
    """
    if not isinstance(X, np.ndarray):
        raise RuntimeError("Wrong input")
    if len(X.shape) == 1:
        return norm(norm(X, ord=p), ord=q)
    elif len(X.shape) == 2:
        return norm(norm(X, axis=1, ord=p), ord=q)
    else:
        raise RuntimeError("Unsupported dimension of X.")


def prox(constrain_type, constrain_lambda, x, C=None):
    def prox_l1(x, l):
        return np.sign(x) * np.maximum(np.abs(x) - l, 0)
    def prox_l21(x, l):
        x_norm = norm(x, axis=1)
        x_norm[x_norm == 0] = 1
        return np.maximum(1-l/x_norm, 0).reshape(-1, 1) * x
    def prox_l21c(x, l, C):
        n_constrain = int(np.sum(C[0]))
        C = C.astype(np.bool)
        x_c = x[C].reshape(-1, n_constrain)
        x_u = x[~C].reshape(-1, x.shape[1] - n_constrain)
        x_c = prox_l21(x_c, l)
        if C[0, 0]:
            return np.hstack([x_c, x_u])
        else:
            return np.hstack([x_u, x_c])

    if constrain_type is None:
        return x
    elif constrain_type == "l1":  # L_1 norm
        return prox_l1(x, constrain_lambda)
    elif constrain_type == 'l21':  # L_{2,1} norm
        return prox_l21(x, constrain_lambda)
    elif constrain_type == 'l21c':  # L_{2,1} norm with indication matrix
        return prox_l21c(x, constrain_lambda, C)
    else:
        raise RuntimeError("Unsupported constrain type")


def g(constrain_type, x, C=None):
    """
    calculate the constrain part of cost function according to the type.
    """
    if constrain_type is None:
        return 0
    elif constrain_type == "l1":  # L1 norm
        return np.sum(np.abs(x))
    elif constrain_type == 'l21':  # L_{2,1} norm
        return mixed_norm(x, 2, 1)
    elif constrain_type == 'l21c':  # L_{2,1} norm with indication matrix C
        return mixed_norm(x*C, 2, 1)


def line_search(f, constrain_type, constrain_lambda, grad, x, step, beta=0.5):
    while True:
        z = prox(constrain_type, constrain_lambda * step, x - step * grad(x))
        cost_hat = f(x) + np.dot(z.ravel() - x.ravel(), grad(x).ravel()) + 1/(2*step)*norm(z - x) + constrain_lambda * g(constrain_type, z)
        if cost_hat >= f(z) + constrain_lambda * g(constrain_type, z):
            break
        step *= beta
    return step, z


def apg(f, constrain_type, constrain_lambda, grad, x_init, constrain_C=None,
        lipschitz=None, step_init=1, loop_tol=1e-6, max_iter=500, verbose=False):
    """
    Accelerated Proximal Gradient Method.
    :param f: cost function of f(x) in min{ f(x) + g(x) }
    :param constrain_type: type of cost function of non-smooth part g(x) in min { f(x) + \lambda g(x) }
    :param constrain_lambda: the coefficient of constrain part.
    :param constrain_C: the indication matrix for constrain, only used when constrain type is l21c
    :param grad: gradient function of f(x).
    :param x_init: the initial value of x
    :param lipschitz: lipschitz constant. if not given, line search method will be exploited.
    :param step: the initial step of line search.
    :param loop_tol: the tolerance of final result
    :param max_iter: maximum number of iterations
    :return: the final calculated value of x.
    """
    x = x_init
    x_old = np.zeros_like(x)   # x_old's initial value doesn't matter
    iter = 0
    if lipschitz is not None:
        step = 1 / lipschitz
    while True:
        omega = iter / (iter + 3)
        y = x + omega * (x - x_old)
        if lipschitz is None:
            step, z = line_search(f, constrain_type, constrain_lambda, grad, y, step=step_init, beta=0.5)
        if constrain_C is not None:
            z = prox(constrain_type, constrain_lambda * step, y - step * grad(y), C=constrain_C)
        else:
            z = prox(constrain_type, constrain_lambda * step, y - step * grad(y))

        x_old = x
        x = z  # update x by z
        if norm(x - x_old) <= loop_tol:
            break
        if iter >= max_iter:
            if lipschitz is None:
                warnings.warn("max_iter exceeded, if the Lipschitz constant is not given, "
                          "consider set it")
            else:
                warnings.warn("max_iter exceeded")
            break
        iter += 1
        if verbose:
            print('Iter: %d\tCurrent x: ' % (iter+1), end='')
            print(x)
    return x


# Below is the testing part
if __name__ == "__main__":
    def test_f1(x):
        return x[0]**2 + 2*x[1]**2 + 3*x[2]**2
    def test_grad1(x):
        return np.array([2*x[0], 4*x[1], 6*x[2]])
    def test_f2(x):
        return np.sum([1, 4, 1, 4] @ x.ravel()**2)
    def test_grad2(x):
        return np.array([[2, 8], [2, 8]]) * x
    def test_f3(x):
        return -200 * np.exp(-0.2 * norm(x))
    def test_grad3(x):
        return 40 * np.exp(-0.2 * norm(x)) / norm(x) * x
    # x_init = np.array([50, 2, 3])
    # x_init = np.array([[1, 100], [20, -43]])
    x_init = np.array([-10, 20])
    x = apg(f=test_f2, constrain_type=None, 
            constrain_lambda=0.1, grad=test_grad2,
            x_init=x_init, lipschitz=10, 
            step_init=10, loop_tol=1e-6, max_iter=20000,
            verbose=True)
