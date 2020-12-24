# SJSPCA

**Note: under developing stage**

Python implementation of structured joint sparse principal component analysis. This implementation is based on the `Transformer` type in `scikit-learn`.

SJSPCA can be used both for fault detection and fault isolation. For fault detection, the $T^2$ and $SPE$ statistic is calculated.

For details, refer to the paper published on IEEE TII: *Structured Joint Sparse Principal Component Analysis for Fault Detection and Isolation*

## Improvement: Lipschitz constant added

In the original paper of SJSPCA, a back-tracking scheme is used to determine the step size of each iteration in the optimization of $\mathbf{B}$. However, as $\nabla f(\mathbf{B})$ is Lipschitz continuous, we can use $1/L$ as a constant step size to guarantee both the convergence and speed. In fact,
$$
\nabla f(\mathbf{B})=2(\mathbf{X}^\top\mathbf{X}+\lambda_2\mathbf{L})\mathbf{B}-2\mathbf{X}^\top\mathbf{X}\mathbf{A}
$$
Too calculate the Hessian, we first vectorize $\mathbf{B}$ and $\nabla f(\mathbf{B})$. Let 
$$
\mathbf{M}=\left[
\begin{matrix}
2(\mathbf{X}^\top\mathbf{X}+\lambda_2\mathbf{L}) & & & \\
& 2(\mathbf{X}^\top\mathbf{X}+\lambda_2\mathbf{L}) & & \\
& & \ddots & \\
& & & 2(\mathbf{X}^\top\mathbf{X}+\lambda_2\mathbf{L})
\end{matrix}
\right]
\in \mathbb{R}^{p^2 \times p^2}
$$
, and
$$
\boldsymbol{b}=\left[\matrix{\boldsymbol{b}_1^\top & \boldsymbol{b}_2^\top &\cdots &\boldsymbol{b}_p^\top}\right]^\top
$$
, where $\boldsymbol{b}_i$ is the ith column of $\mathbf{B}$, then $\nabla f(\mathbf{B})$ can be expressed equivalently as
$$
\nabla f(\mathbf{\boldsymbol{b}}) = \mathbf{M}\boldsymbol{b}+\mathbf{Const.}
$$
therefore
$$
\frac{\part \nabla f(\boldsymbol{b})}{\part \boldsymbol{b}^\top} = \mathbf{M}
$$
which gives
$$
\left\|\frac{\part \nabla f(\boldsymbol{b})}{\part \boldsymbol{b}^\top}\right\| = \|\mathbf{M}\| = 2\sqrt{p}\|(\mathbf{X}^\top\mathbf{X}+\lambda_2\mathbf{L})\|\le2\sqrt{p}\lambda_{max}(\mathbf{X}^\top\mathbf{X}+\lambda_2\mathbf{L})
$$
Thus the Lipschitz constant
$$
L = 2\sqrt{p}\lambda_{max}(\mathbf{X}^\top\mathbf{X}+\lambda_2\mathbf{L})
$$
To further assure convergence, we can set in practice the twice the computed value above as the constant.