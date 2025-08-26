from typing import Callable,Optional
from jaxtyping import PyTree,Array
import jax
import jax.numpy as jnp 




def minres(A_func: Callable, b: PyTree, tol: float = 3e-4, x0: Optional[PyTree] = None, maxiter: int = 100) -> PyTree:
    """
    Simplified MINRES for your G matrix system with PyTree support.
    """
    @jax.jit
    def dot_tree(x: PyTree, y: PyTree) -> Array:
        return sum(jax.tree.leaves(jax.tree.map(lambda a, b: jnp.sum(a * b), x, y)))

    @jax.jit
    def norm_tree(x: PyTree) -> Array:
        return jnp.sqrt(dot_tree(x, x))

    @jax.jit
    def clone_tree(x: PyTree) -> PyTree:
        return jax.tree.map(lambda a: jnp.array(a), x)
    @jax.jit
    def xpay_tree(x: PyTree, y: PyTree, alpha: float) -> PyTree:
        return jax.tree.map(lambda a, b: a + alpha * b, x, y)
    
    # def step():

    if x0 is None:
        x = jax.tree.map(jnp.zeros_like, b)
    else:
        x = clone_tree(x0)

    Ax = A_func(x)
    r = xpay_tree(b, Ax, -1.0)
    p0 = clone_tree(r)
    s0 = A_func(p0)
    p1 = clone_tree(p0)
    s1 = clone_tree(s0)

    for i in range(maxiter):
        
        p2 = clone_tree(p1)
        p1 = clone_tree(p0)
        s2 = clone_tree(s1)
        s1 = clone_tree(s0)

        alpha = dot_tree(r, s1) / dot_tree(s1, s1)

        x = xpay_tree(x, p1, alpha)
        r = xpay_tree(r, s1, -alpha)

        if norm_tree(r) < tol**2:
            print(f"Converged in {i} iterations")
            info = {"success": True, "iterations": i, "norm_res": norm_tree(r)}
            break

        p0 = clone_tree(s1)
        s0 = A_func(s1)
        beta1 = dot_tree(s0, s1) / dot_tree(s1, s1)
        p0 = xpay_tree(p0, p1, -beta1)
        s0 = xpay_tree(s0, s1, -beta1)

        if i > 1:
            beta2 = dot_tree(s0,s2)/dot_tree(s2,s2)
            p0 = xpay_tree(p0, p2, -beta2)
            s0 = xpay_tree(s0, s2, -beta2)
    if i == maxiter - 1:
        info = {"success": False, "iterations": maxiter, "norm_res": norm_tree(r)}
    print(info)
    return x, info

    