import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from flax import nnx
from jaxtyping import Array,PyTree
from typing import Tuple,List,Dict,Optional
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jla
import matplotlib.pyplot as plt
from jax import Device

from geometry.G_matrix import G_matrix
from geometry.lin_alg_solvers import minres
from flows.gradient_flow_step import gradient_flow_step

from functionals.functional import Potential
from parametric_model.parametric_model import ParametricModel

def anderson_step(
        parametric_model: ParametricModel,
        current_params: PyTree,
        param_history: List[PyTree],  # at most m entries
        residual_history: List[PyTree],  # at most m entries
        G_mat: G_matrix,
        potential: Potential,
        z_samples: Array,
        step_size: float = 0.01,
        memory_size: int = 5,
        mixing_parameter: float = 1.0,
        anderson_tol: float = 1e-6,
        solver: str = "minres",
        solver_tol: float = 1e-6,
        solver_maxiter: int = 50,
        regularization: float = 1e-6
    ) -> Tuple[PyTree, List[PyTree], List[PyTree], Dict]:
    '''
    Anderson acceleration step for fixed-point iteration.
    Args:
        parametric_model: ParametricModel instance
        current_params: Current parameters of the model
        param_history: List of previous parameters (at most memory_size entries)
        residual_history: List of previous residuals (at most memory_size entries)
        G_mat: G_matrix object to compute inner products
        potential: Potential object to compute energy and gradient
        z_samples: Reference samples (batch_size, d)
        device: JAX device to perform computations on
        step_size: Step size for the fixed-point iteration
        memory_size: Number of previous iterations to use for Anderson acceleration
        mixing_parameter: Mixing parameter for Anderson acceleration
        anderson_tol: Tolerance for Anderson acceleration convergence
        solver: Linear solver to use ('minres' or 'cg')
        solver_tol: Tolerance for the linear solver
        regularization: Regularization parameter for the linear system
    Returns:
        new_params: Updated parameters after Anderson acceleration
        new_param_history: Updated parameter history
        new_residual_history: Updated residual history
        info: Dictionary with information about the step (e.g., energy, gradient norm)
    '''
    
    hist_len = len(param_history)

    if hist_len == 0:
        # No history, perform a standard gradient flow step
        new_params, _ = gradient_flow_step(
            parametric_model= parametric_model,
            z_samples = z_samples,
            G_mat = G_mat,
            potential = potential,
            step_size=step_size,
            solver=solver,
            solver_tol=solver_tol,
            solver_maxiter=50,
            regularization=regularization,
            only_return_params=True
        )
        current_residual = compute_fixed_point_residual(
            parametric_model, current_params, G_mat, potential, z_samples,
            step_size, solver, solver_tol, regularization
        )
        return new_params, [current_params], [current_residual], {}
    
    # Compute fixed-point residuals for current parameter
    current_residual = compute_fixed_point_residual(
        parametric_model, current_params, G_mat, potential, z_samples,
        step_size, solver, solver_tol, regularization
    )
    # Build difference matrices 
    m_k = min(memory_size, hist_len)
    param_hist_trunc = param_history[-m_k:]
    residual_hist_trunc = residual_history[-m_k:]

    param_diffs = []
    residual_diffs = []

    all_params = param_hist_trunc + [current_params]
    all_residuals = residual_hist_trunc + [current_residual]

    for i in range(m_k):
        delta_parm = jax.tree.map(lambda a,b: a-b,all_params[i+1], all_params[i])
        delta_res = jax.tree.map(lambda a,b: a-b,all_residuals[i+1], all_residuals[i])
        param_diffs.append(delta_parm)
        residual_diffs.append(delta_res)
    
    # Compute Anderson mixing coefficients
    gamma, anderson_info = compute_anderson_gamma(
        current_residual, residual_diffs, G_mat, z_samples, tol=anderson_tol
    )   
    # Compute new parameter estimate
    mixed_residual = current_residual
    for i,gamma_i in enumerate(gamma):
        mixed_residual = jax.tree.map(lambda a,b: a -gamma_i * b, mixed_residual, residual_diffs[i])
    
    delta_theta = jax.tree.map(lambda x: mixing_parameter * x, mixed_residual)
    for i,gamma_i in enumerate(gamma):
        delta_theta = jax.tree.map(lambda step,dx: step -gamma_i * dx, delta_theta, param_diffs[i])
    new_params = jax.tree.map(lambda p, d: p + d, current_params, delta_theta)

    updated_param_history = param_hist_trunc + [current_params]
    updated_residual_history = residual_hist_trunc + [current_residual]
    return new_params, updated_param_history, updated_residual_history 



    
def compute_fixed_point_residual(
    parametric_model: nnx.Module,
        params: PyTree,
        G_mat: G_matrix, 
        potential: Potential,
        z_samples: Array,
        step_size: float,
        solver: str,
        solver_tol: float,
        regularization: float
    ) -> PyTree:
    '''
    Compute fixed point residual r = -h * G^{-1} grad F(p) for parameters p
    Args:
        parametric_model: Neural ODE model
        params: Current parameters of the model
        G_mat: G_matrix object to compute inner products
        potential: Potential object to compute energy and gradient
        z_samples: Reference samples (batch_size, d)
        step_size: Step size for the fixed-point iteration
        solver: Linear solver to use ('minres' or 'cg')
        solver_tol: Tolerance for the linear solver
        regularization: Regularization parameter for the linear system
    Returns:
        residual: Fixed point residual as a PyTree
    '''

    # Compute energy gradient using the potential
    energy_grad,energy,energy_breakdown = potential.compute_energy_gradient(parametric_model, z_samples, params)
    # Solve linear system
    eta, solver_info = G_mat.solve_system(z_samples, energy_grad,
                                            params=params,
                                            tol=solver_tol, 
                                            maxiter=50,
                                            method=solver,
                                            regularization=regularization)
    # Fixed point residual
    residual = jax.tree.map(lambda x: -step_size * x, eta)
    return residual


def compute_anderson_gamma(
    current_residual: PyTree,
    residual_differences: List[PyTree], 
    G_mat: G_matrix,
    z_samples: Array,
    tol: float = 1e-6
) -> Tuple[List[float], Dict]:
    """
    Solve Anderson mixing optimization using G-matrix norm:
    Args:
        current_residual: Current fixed-point residual r_k
        residual_differences: List of previous residual differences (r_{i+1} - r_i)
        G_mat: G_matrix object to compute inner products
        z_samples: Reference samples (batch_size, d)
        tol: Tolerance for the linear solver   
    Returns:
        gamma: Coefficients for Anderson mixing
        info: Dictionary with solver information
    """
    m = len(residual_differences)
    
    if m == 0:
        return [], {'converged': True, 'residual_reduction': 0.0}
    
    # Build least-squares system
    A = jnp.zeros((m, m))
    b = jnp.zeros((m,))

    for i in range(m):
        for j in range(m):
            A[i, j] = G_mat.inner_product(residual_differences[i], residual_differences[j], z_samples)
        b[i] = G_mat.inner_product(current_residual, residual_differences[i], z_samples)
    
    # Solve the linear system A gamma = b
    # try:
    #     gamma, info = jla.solve(A + tol * jnp.eye(m), b, sym_pos=True, assume_a='pos', overwrite_a=False, overwrite_b=False, check_finite=False)
    #     converged = True
    # except jla.LinAlgError:
    #     gamma = jnp.linalg.pinv(A + tol * jnp.eye(m)) @ b
    #     converged = False
    #     info = {}
    gamma,info = minres(
        A_func=lambda x: jnp.dot(A, x),
        b=b,
        tol=tol,
        maxiter=100
    )
    converged = info.get('success', False)

    return gamma.tolist(), {'converged': converged}
    