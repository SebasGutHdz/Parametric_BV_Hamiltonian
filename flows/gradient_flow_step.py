import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from flax import nnx
from jaxtyping import Array,PyTree
from typing import Tuple,Any,Union
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import jax
from jax import Device

from geometry.G_matrix import G_matrix

from functionals.functional import Potential

from tqdm import tqdm

def move_to_device(pytree: Any, device) -> Any:
    """Recursively moves all JAX arrays in a PyTree to the specified device."""
    return jax.tree.map(lambda x: jax.device_put(x, device) if isinstance(x, jax.Array) else x, pytree)


def gradient_flow_step(node: nnx.Module, z_samples: Array, G_mat: G_matrix,
                                    potential: Potential, step_size: float = 0.01,
                                    solver: str = "minres", solver_tol: float = 1e-6,
                                    solver_maxiter: int = 50,regularization: float = 1e-6,only_return_params: bool = False) -> Tuple[Union[nnx.Module,PyTree], dict]:
    """
    Generic gradient flow step that works with any Potential
    
    Args:
        node: Current Neural ODE model
        z_samples: Reference samples for Monte Carlo estimation
        G_mat: G-matrix object for linear system solving
        potential: Potential instance
        step_size: Gradient flow step size h TODO: Implement higher order solvers or adaptive step sizing
        solver_tol: Tolerance for linear solver
        solver_maxiter: Maximum iterations for linear solver
        regularization: Regularization parameter used in regularized cg
    Returns:
        updated_node: Node with updated parameters
        step_info: Dictionary with step diagnostics
    """
    
    # Get current parameters
    _, current_params = nnx.split(node)
    
    # Compute energy gradient using the potential
    energy_grad,energy,energy_breakdown = potential.compute_energy_gradient(node, z_samples, current_params)
    
    # Solve linear system   
    # z_samples_g_mat = z_samples[::2]  # Use a subset of samples for G-matrix to save computation

    eta, solver_info = G_mat.solve_system(z_samples, energy_grad,
                                            params=current_params,
                                            tol=solver_tol, 
                                            maxiter=solver_maxiter,
                                            method=solver,
                                            regularization=regularization)
    # ODE solve. 
    # TODO: Higher order derivative solvers. 
    updated_params = jax.tree.map(lambda p, e: p - step_size * e, current_params, eta)

    if only_return_params:
        return updated_params, {}
    
    # Create updated node
    graphdef, _ = nnx.split(node)
    updated_node = nnx.merge(graphdef, updated_params)
    # updated_node = move_to_device(updated_node, device)
    # Compute diagnostics
    grad_norm = jnp.sqrt(sum(jax.tree.leaves(jax.tree.map(lambda x: jnp.sum(x**2), energy_grad))))
    eta_norm = jnp.sqrt(sum(jax.tree.leaves(jax.tree.map(lambda x: jnp.sum(x**2), eta))))
    param_norm = jnp.sqrt(sum(jax.tree.leaves(jax.tree.map(lambda x: jnp.sum(x**2), updated_params))))

    step_info = {
        'gradient_norm': grad_norm,
        'eta_norm': eta_norm,
        'param_norm': param_norm,
        'energy': energy,
        'internal_energy': energy_breakdown['internal_energy'],
        'linear_energy': energy_breakdown['linear_energy'],
        'interaction_energy': energy_breakdown['interaction_energy'],
        'step_size': step_size
    }
    
    return updated_node, step_info