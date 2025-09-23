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


def hamiltonian_flow_step(node: nnx.Module, z_samples: Array, G_mat: G_matrix,
                                    potential: Potential, step_size: float = 0.01,
                                    solver: str = "minres", solver_tol: float = 1e-6,
                                    solver_maxiter: int = 50,regularization: float = 1e-6,only_return_params: bool = False) -> Tuple[Union[nnx.Module,PyTree], dict]:
    """
    Generic hamiltonian flow step that works with any Potential
    
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







    return 