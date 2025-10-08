import os
import sys
# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from typing import Callable,Optional,Any,List,Tuple
from jaxtyping import PyTree,Array
import jax
import jax.numpy as jnp 
from jax.scipy.sparse.linalg import cg
from jax import lax


def reg_cg(A_func: Callable, b: PyTree,epsilon: float =  1e-6, tol: float = 1e-6, x0: Optional[PyTree] = None, maxiter: int = 100) -> tuple[PyTree, dict]:

    def regu_A(x: PyTree) -> PyTree:
        return jax.tree.map(lambda x,y: x+epsilon*y, A_func(x), x)


    return cg(regu_A, b, x0=x0, tol=tol, maxiter=maxiter)


# def minres(A_func: Callable, b: PyTree, tol: float = 1e-6, x0: Optional[PyTree] = None, maxiter: int = 100) -> tuple[PyTree, dict]:
#     """
#     MINRES implementation for PyTree support.
#     """
    
#     @jax.jit
#     def dot_tree(x: PyTree, y: PyTree) -> Array:
#         return sum(jax.tree.leaves(jax.tree.map(lambda a, b: jnp.sum(a * b), x, y)))

#     @jax.jit
#     def norm_tree(x: PyTree) -> Array:
#         return jnp.sqrt(dot_tree(x, x))

#     @jax.jit
#     def scale_tree(x: PyTree, alpha: float) -> PyTree:
#         return jax.tree.map(lambda a: alpha * a, x)
    
#     @jax.jit
#     def add_trees(x: PyTree, y: PyTree) -> PyTree:
#         return jax.tree.map(lambda a, b: a + b, x, y)
    
#     @jax.jit
#     def sub_trees(x: PyTree, y: PyTree) -> PyTree:
#         return jax.tree.map(lambda a, b: a - b, x, y)

#     # Initialize
#     if x0 is None:
#         x = jax.tree.map(jnp.zeros_like, b)
#     else:
#         x = jax.tree.map(lambda a: jnp.array(a), x0)

#     # Initial residual
#     Ax = A_func(x)
#     r = sub_trees(b, Ax)
    
#     # MINRES initialization
#     v_old = jax.tree.map(jnp.zeros_like, b)
#     v = jax.tree.map(lambda a: jnp.array(a), r)
#     w_old = jax.tree.map(jnp.zeros_like, b)
#     w = jax.tree.map(jnp.zeros_like, b)
    
#     beta = norm_tree(r)
#     initial_residual = beta
    
#     if beta < tol:
#         info = {"success": True, "iterations": 0, "norm_res": beta}
#         return x, info
    
#     # Normalize v
#     v = scale_tree(v, 1.0 / beta)
    
#     eta = beta
#     s_old = 0.0
#     c_old = 1.0
    
#     for i in range(maxiter):
#         # Lanczos process
#         Av = A_func(v)
        
#         if i > 0:
#             Av = sub_trees(Av, scale_tree(v_old, beta))
        
#         alpha = dot_tree(v, Av)
#         Av = sub_trees(Av, scale_tree(v, alpha))
        
#         beta = norm_tree(Av)
        
#         # Apply previous Givens rotation
#         if i > 0:
#             delta = c_old * alpha + s_old * beta
#             gamma_bar = s_old * alpha - c_old * beta
#         else:
#             delta = alpha
#             gamma_bar = beta
        
#         # Compute new Givens rotation
#         if abs(gamma_bar) < 1e-14:  # Avoid division by zero
#             c = 1.0
#             s = 0.0
#             gamma = delta
#         else:
#             if abs(gamma_bar) > abs(delta):
#                 tau = delta / gamma_bar
#                 s = 1.0 / jnp.sqrt(1.0 + tau**2)
#                 c = s * tau
#             else:
#                 tau = gamma_bar / delta
#                 c = 1.0 / jnp.sqrt(1.0 + tau**2)
#                 s = c * tau
#             gamma = c * delta + s * gamma_bar
        
#         # Update solution
#         if abs(gamma) > 1e-14:  # Avoid division by zero
#             eta_new = -s * eta
#             eta = c * eta
            
#             w_new = sub_trees(sub_trees(v, scale_tree(w_old, gamma_bar)), scale_tree(w, delta))
#             w_new = scale_tree(w_new, 1.0 / gamma)
            
#             x = add_trees(x, scale_tree(w_new, eta))
            
#             # Update for next iteration
#             w_old = jax.tree.map(lambda a: jnp.array(a), w)
#             w = jax.tree.map(lambda a: jnp.array(a), w_new)
#             eta = eta_new
        
#         # Check convergence
#         residual_norm = abs(eta)
        
#         if residual_norm < tol:
#             info = {"success": True, "iterations": i + 1, "norm_res": residual_norm}
#             return x, info
        
#         # Prepare for next iteration
#         if beta > 1e-14 and i < maxiter - 1:
#             v_old = jax.tree.map(lambda a: jnp.array(a), v)
#             v = scale_tree(Av, 1.0 / beta)
#             c_old = c
#             s_old = s
#         else:
#             break
    
#     # Did not converge
#     info = {"success": False, "iterations": maxiter, "norm_res": residual_norm}
#     return x, info

def dot_tree(x: PyTree, y: PyTree) -> Array:
    xl = jax.tree.leaves(x)
    yl = jax.tree.leaves(y)
    return sum(jnp.vdot(a, b) for a, b in zip(xl, yl))

def norm_tree(x: PyTree) -> Array:
    return jnp.sqrt(jnp.real(dot_tree(x, x)))

def scale_tree(x: PyTree, alpha: Array) -> PyTree:
    return jax.tree.map(lambda a: alpha * a, x)

def add_trees(x: PyTree, y: PyTree) -> PyTree:
    return jax.tree.map(lambda a, b: a + b, x, y)

def zeros_like_tree(x: PyTree) -> PyTree:
    return jax.tree.map(jnp.zeros_like, x)

def linear_combination(vs: List[PyTree], coeffs: Array) -> PyTree:
    """Compute sum_j coeffs[j] * vs[j] where vs is list of PyTrees and coeffs is 1D array."""
    out = zeros_like_tree(vs[0])
    for j, vj in enumerate(vs):
        out = add_trees(out, scale_tree(vj, coeffs[j]))
    return out

def minres(
    A_func: Callable[[PyTree], PyTree],
    b: PyTree,
    tol: float = 1e-6,
    x0: Optional[PyTree] = None,
    maxiter: int = 100,
) -> Tuple[PyTree, dict]:
    """
    MINRES for symmetric (Hermitian) A with PyTree vectors.
    This version builds Lanczos vectors and solves the small least-squares problem
    (clear and robust reference implementation).
    """
    # initial guess
    if x0 is None:
        x = zeros_like_tree(b)
    else:
        x = jax.tree.map(lambda a: jnp.asarray(a), x0)

    # initial residual r0 = b - A x
    Ax = A_func(x)
    r0 = jax.tree.map(lambda a, bb: bb - a, Ax, b)  # b - Ax

    beta1 = float(norm_tree(r0))
    initial_residual = beta1

    if beta1 == 0.0:
        return x, {"success": True, "iterations": 0, "norm_res": 0.0}

    # Lanczos basis
    v1 = scale_tree(r0, 1.0 / beta1)
    Vs: List[PyTree] = [v1]

    alphas: List[float] = []
    betas: List[float] = []

    xk = x  # placeholder so xk is defined if loop doesn't run

    for k in range(1, maxiter + 1):
        v = Vs[-1]
        Av = A_func(v)

        # alpha_k = v^T A v
        alpha = jnp.real(dot_tree(v, Av))
        alphas.append(float(alpha))

        # w = Av - alpha * v - beta_{k-1} * v_{k-1}
        w = jax.tree.map(lambda a, b: a - alpha * b, Av, v)
        if k > 1:
            beta_prev = betas[-1]
            v_prev = Vs[-2]
            w = jax.tree.map(lambda a, b: a - beta_prev * b, w, v_prev)

        beta_k = float(norm_tree(w))
        betas.append(beta_k)

        if beta_k == 0.0:
            # Lanczos terminated exactly; do not attempt to form v_{k+1}
            pass
        else:
            v_next = scale_tree(w, 1.0 / beta_k)
            Vs.append(v_next)

        # Build (k+1) x k tridiagonal Tk_under (T_k with extra bottom row)
        Tk_under = jnp.zeros((k + 1, k), dtype=jnp.float64)
        for j in range(k):
            Tk_under = Tk_under.at[j, j].set(alphas[j])
            # sub-diagonal: Tk_under[j+1, j] = beta_{j}  (betas indexed 0..k-1)
            Tk_under = Tk_under.at[j + 1, j].set(betas[j])

        # rhs = beta1 * e1
        rhs = jnp.zeros((k + 1,), dtype=jnp.float64).at[0].set(beta1)

        # Solve least-squares via reduced QR (robust and avoids jnp.linalg.lstsq availability issues)
        # Tk_under is (k+1, k) -> Q:(k+1,k), R:(k,k)
        Q, R = jnp.linalg.qr(Tk_under, mode="reduced")
        # Solve R y = Q^T rhs
        y = jnp.linalg.solve(R, Q.T @ rhs)

        # form x_k = V_k * y (only first k vectors)
        y = jnp.asarray(y)[:k]
        xk = linear_combination(Vs[:k], y)

        # compute residual norm || rhs - Tk_under @ y ||
        res_vec = rhs - Tk_under @ y
        res_norm = float(jnp.linalg.norm(res_vec))

        if res_norm <= tol:
            return xk, {"success": True, "iterations": k, "norm_res": res_norm}

        # if Lanczos terminated exactly (beta_k == 0) we cannot continue further
        if beta_k == 0.0:
            return xk, {"success": False, "iterations": k, "norm_res": res_norm}

    # reached maxiter
    return xk, {"success": False, "iterations": maxiter, "norm_res": res_norm}