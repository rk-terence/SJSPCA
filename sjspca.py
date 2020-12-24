from warnings import warn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_is_fitted
import numpy as np
from numpy.linalg import norm
from apgd import apg


class SJSPCA(BaseEstimator, TransformerMixin):
    def __init__(self, n_components, lambda1, lambda2, knn_sigma, residual_penalty=True, tol=1e-5, max_iter=200):
        self.n_components = n_components
        self.l1 = lambda1
        self.l2 = lambda2
        self.tol = tol
        self.max_iter = max_iter
        self.knn_sigma = knn_sigma
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
    
    def _update_A(X, B):
        U, S, Vh = np.linalg.svd((X.T @ X) @ B)
        U = U[:, :S.shape[0]]
        Vh = Vh[:S.shape[0]]
        return U @ Vh
        
    def fit(self, X, y=None, A_init=None, L=None):
        n_features = X.shape[1]
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
            L = self._gen_laplace(X, self.knn_sigma)
        k = 0
        B_old = np.eye(n_features)
        inner_tol = 0.1 * self.tol
        while True:
            # solve B with A fixed
            Lc = self._compute_lipschitz(X, L)
            f = lambda B: self._f(X, A_old, B, L)
            grad = lambda B: self._grad(X, A_old, B, L)
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
        self.T_ = X @ self.B_
        self.Lambda_ = np.var(self.T_, axis=0)  # the variance of each PC
        return A, B
    
    def transform(self, X, y=None):
        check_is_fitted(self)  # to check if SJSPCA is fitted
        return X @ self.B_
    
    def compute_t2(self, X):
        check_is_fitted(self)
        return X @ B
    

if __name__ == "__main__":
    print("sjspca.py")