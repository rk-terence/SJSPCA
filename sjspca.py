from warnings import warn
import copy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_is_fitted
import numpy as np
from numpy.linalg import norm
from apgd import apg


class SJSPCA(BaseEstimator, TransformerMixin):
    def __init__(self, n_components, l1, l2, knn_sigma, knn_n_neighbors=None, 
                 partial_penalty=False, residual_penalty=True, 
                 tol=1e-5, max_iter=200):
        self.n_components = n_components
        self.l1 = l1
        self.l2 = l2
        self.tol = tol
        self.max_iter = max_iter
        self.knn_sigma = knn_sigma
        self.knn_n_neighbors = knn_n_neighbors
        self.partial_penalty = partial_penalty
        self.residual_penalty = residual_penalty
    
    @staticmethod
    def _gen_laplace(X, n_neighbors=None, sigma=None):
        if n_neighbors is None:
            n_neighbors = X.shape[1]
        neigh = NearestNeighbors(n_neighbors=n_neighbors).fit(X.T)
        graph = np.array(neigh.kneighbors_graph(X.T, mode='distance').todense())

        if sigma is None:
            graph_ = graph[graph < np.inf].reshape(X.shape[1], n_neighbors)
            sigma = np.average(np.var(graph_, axis=1))  # sigma is the mean of std of all weights

        W = np.exp(-1 * np.power(graph, 2) / sigma)
        D = np.diag(np.sum(W, axis=1))
        L = D - W

        return L, W
    
    def _f(self, X, A, B, L):
        return norm(X - X @ B @ A.T)**2 + self.l2 * np.trace(B.T @ L @ B)
    
    def _grad(self, X, A, B, L):
        return 2 * X.T @ X @ (B - A) + 2 * self.l2 * L @ B
    
    def _compute_lipschitz(self, X, L):
        return 2 * np.sqrt(X.shape[1]) * np.linalg.eig(X.T @ X + self.l2 * L)[0].max()
    
    @staticmethod
    def _update_A(X, B):
        U, S, Vh = np.linalg.svd((X.T @ X) @ B)
        U = U[:, :S.shape[0]]
        Vh = Vh[:S.shape[0]]
        return U @ Vh
        
    def fit(self, X, y=None, A_init=None, L=None):
        n_features = X.shape[1]
        if self.partial_penalty:
            if self.residual_penalty:  # apply L_{2, 1} norm on the residual space
                C = np.hstack([np.zeros([n_features, self.n_components]),
                            np.ones([n_features, n_features - self.n_components])])
            else:
                C = np.hstack([np.ones([n_features, self.n_components]),
                            np.zeros([n_features, n_features - self.n_components])])
        if A_init is None:
            A_old = np.eye(n_features)
        else:
            A_old = A_init
        if L is None:
            L, _ = self._gen_laplace(X, n_neighbors=self.knn_n_neighbors, sigma=self.knn_sigma)
        k = 0
        B_old = np.eye(n_features)
        inner_tol = 0.1 * self.tol
        Lc = self._compute_lipschitz(X, L)
        while True:
            # solve B with A fixed
            f = lambda B: self._f(X, A_old, B, L)
            grad = lambda B: self._grad(X, A_old, B, L)
            if self.partial_penalty:
                B = apg(f=f, constrain_type='l21c', constrain_lambda=self.l1, constrain_C=C, grad=grad,
                        x_init=B_old, lipschitz=Lc, loop_tol=inner_tol)
            else:
                B = apg(f=f, constrain_type='l21', constrain_lambda=self.l1, grad=grad,
                        x_init=B_old, lipschitz=Lc, loop_tol=inner_tol)
            # solve A with B fixed
            A = self._update_A(X, B)
            
            if norm(B - B_old)**2  < self.tol:
                break
            
            k += 1
            B_old = B
            A_old = A
            
            if k >= self.max_iter:
                warn("max_iter exceeded!\n",
                     "norm(self.B - B_old):", norm(B - B_old))
                break
        self.L_ = L.copy()
        self.A_ = A.copy()
        self.B_ = B.copy()
        self.Br_ = B[:, :self.n_components].copy()
        self.Bd_ = B[:, self.n_components:].copy()
        self.T_ = X @ self.B_
        self.Lambda_ = np.var(self.T_, axis=0)  # the variance of each PC
        return self
    
    def transform(self, X, y=None):
        check_is_fitted(self)  # to check if SJSPCA is fitted
        return X @ self.B_
    
    def compute_t2(self, X):
        check_is_fitted(self)
        t2 = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            t2[i] = X[i] @ self.Br_ @ np.diag(1 / self.Lambda_) @ self.Br_.T @ X[i]
        return t2
            
    def compute_spe(self, X):
        check_is_fitted(self)
        spe = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            spe[i] = X[i] @ (np.eye(X.shape[1]) - self.Br_ @ self.Br_.T) @ X[i]
        return spe


def shrinkage(sjspca, X, l_init=1, step=2):
    l1 = l_init
    sjspca = copy.deepcopy(sjspca)
    sjspca.set_params(partial_penalty=True, residual_penalty=True, n_components=2)
    while True:
        sjspca.set_params(l1=l1)
        Bd = sjspca.fit(X).Bd_
        if Bd.all():
            break
        l1 *= step
    Br = sjspca.Br_
    A = sjspca.A_
    return l1, Br, A


def isolation(l1, Br, A, l2, Xf, L, verbose=True):
    n_r = Br.shape[1]
    n_d = A.shape[1] - n_r
    Ar = A[:, :n_r]
    Ad = A[:, n_r:]
    Xstar = Xf - Xf @ Br @ Ar.T
    f = lambda dB : norm(Xstar - Xf @ dB @ Ad.T) + l2*(
        np.trace(Br.T @ L @ Br) + np.trace(dB.T @ L @ dB))
    grad = lambda dB : 2 * Xf.T @ Xf @ dB + 2 * l2 @ L @ dB - 2 * Xf.T @ Xstar @ Ad
    Lc = 2 * np.sqrt(n_d) * np.linalg.eig(Xf.T @ Xf + l2 * L)[0].max()
    dBinit = np.zeros_like(Ad)
    dB = apg(f=f, constrain_type='l21', constrain_lambda=l1, grad=grad, x_init=dBinit, lipschitz=Lc)
    fault_score = norm(dB, ord=1, axis=1) / n_d
    if verbose:
        print('Fault score:')
        print(fault_score)
    faulty_variables = np.nonzero(fault_score)[0]
    return faulty_variables
    

if __name__ == "__main__":
    print("testing sjspca.py...\nusing the simulation example in SJSPCA paper\n")
    print("#1 normal SJSPCA")
    H = np.random.multivariate_normal(np.zeros(2), np.diag([0.98, 1]), size=500)
    S = np.array([[1, 1, 1, 1, 0, 0, 0, 0, -0.6, -0.6],
                [0, 0, 0, 0, 1, 1, 1, 1, 0.8, 0.8]])
    Sh = S.T
    e = np.random.multivariate_normal(np.zeros(10), np.diag([0.09, 0.09, 0.09, 0.09,
                                                            0.16, 0.16, 0.16, 0.16,
                                                            0.25, 0.25]), size=500)
    X = H @ S + e
    l1 = 220
    l2 = 50
    sigma = 0.002
    n_neighbors = 10
    sjspca = SJSPCA(n_components=2, l1=l1, l2=l2, 
                    knn_sigma=sigma, knn_n_neighbors=n_neighbors, residual_penalty=False)
    sjspca.fit(X)
    print(sjspca.B_)
    
    print("#2 SJSPCA for fault detection and isolation")
    Bias = np.zeros([500, 10])
    Bias[150:, [1, 3]] = [1.2, 1.8]
    Xf = X + Bias
    
    l1, Br, A = shrinkage(sjspca, X)
    print('l1:', l1)
    print('Br:')
    print(Br)
    print('A:')
    print(A)
    faulty_idx = isolation(l1, Br, A, l2, Xf=Xf, L=sjspca.L_, verbose=True)
