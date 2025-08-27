import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from flax import nnx
from jaxtyping import Array,PyTree
from typing import Tuple
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import jax
from jax import Device
from geometry.G_matrix import G_matrix
from functionals.linear_funcitonal_class import LinearPotential





def gradient_flow_step(node: nnx.Module, z_samples: Array, G_mat: G_matrix,
                                    potential: LinearPotential, step_size: float = 0.01,
                                    solver: str = "minres", solver_tol: float = 1e-6,
                                    solver_maxiter: int = 50,regularization: float = 1e-6) -> Tuple[nnx.Module, dict]:
    """
    Generic gradient flow step that works with any LinearPotential
    
    Args:
        node: Current Neural ODE model
        z_samples: Reference samples for Monte Carlo estimation
        G_mat: G-matrix object for linear system solving
        potential: LinearPotential instance
        step_size: Gradient flow step size h
        solver_tol: Tolerance for linear solver
        solver_maxiter: Maximum iterations for linear solver
        use_regularization: Whether to use regularized CG solver
        
    Returns:
        updated_node: Node with updated parameters
        step_info: Dictionary with step diagnostics
    """
    
    # Get current parameters
    _, current_params = nnx.split(node)
    
    # Compute energy gradient using the potential
    energy_grad = potential.compute_energy_gradient(node, z_samples, current_params)
    
    # Solve linear system: G(θ) η = -∇_θ F(θ)
    neg_energy_grad = jax.tree.map(lambda x: -x, energy_grad)
    
    
    eta, solver_info = G_mat.solve_system(z_samples, neg_energy_grad,
                                            params=current_params,
                                            tol=solver_tol, 
                                            maxiter=solver_maxiter,
                                            method=solver,
                                            regularization=regularization)
    # Update parameters: θ^{k+1} = θ^k + h * η
    updated_params = jax.tree.map(lambda p, e: p + step_size * e, current_params, eta)
    
    # Create updated node
    graphdef, _ = nnx.split(node)
    updated_node = nnx.merge(graphdef, updated_params)
    
    # Compute diagnostics
    grad_norm = jnp.sqrt(sum(jax.tree.leaves(jax.tree.map(lambda x: jnp.sum(x**2), energy_grad))))
    eta_norm = jnp.sqrt(sum(jax.tree.leaves(jax.tree.map(lambda x: jnp.sum(x**2), eta))))
    param_norm = jnp.sqrt(sum(jax.tree.leaves(jax.tree.map(lambda x: jnp.sum(x**2), updated_params))))

    step_info = {
        'gradient_norm': grad_norm,
        'eta_norm': eta_norm,
        'param_norm': param_norm,
        'step_size': step_size
    }
    
    return updated_node, step_info


# Gradient Flow Integration with LinearPotential

def run_gradient_flow(node: nnx.Module, z_samples: Array, G_mat: G_matrix,
                     potential: LinearPotential, device_idx: int = 0, h: float = 0.01, solver: str = "minres",
                     max_iterations: int = 100, tolerance: float = 1e-6,
                     regularization: float = 1e-6, 
                     progress_every: int = 10) -> dict:
    """
    Run complete gradient flow integration with any LinearPotential
    
    Args:
        node: Initial Neural ODE model
        z_samples: Reference samples for Monte Carlo estimation
        G_mat: G-matrix object for linear system solving
        potential: LinearPotential instance defining the energy functional
        solver: str type of solver, choose from cg, and minres
        h: Time step size
        max_iterations: Maximum number of gradient flow steps
        tolerance: Convergence tolerance for energy
        use_regularization: Whether to use regularized CG solver
        progress_every: Print progress every N iterations
        
    Returns:
        results: Dictionary containing energy history, solver stats, etc.
    """
    

    print(f"Starting gradient flow with {potential.__class__.__name__}...")
    print(f"Potential function: {potential.potential_fn.__name__}")
    print(f"Potential parameters: {potential.potential_kwargs}")

    device = jax.devices()[device_idx] if jax.devices() else jax.devices('cpu')[0] # If only one gpu change index to 0
    print(f"Selected device: {device}")
    
    # Initialize tracking
    energy_history = []
    solver_stats = []
    param_norms = []
    sample_history = []  # Store samples at key iterations for visualization
    
    # Initial state
    _, current_params = nnx.split(node)
    current_energy, current_samples = potential.evaluate_energy(node, z_samples, current_params)
    energy_history.append(float(current_energy))
    sample_history.append(current_samples)
    
    print(f"Initial energy: {current_energy:.6f}")
    print(f"Target: minimize energy functional")
    
    # Main integration loop
    current_node = node

    #Initialize key for sample generation
    key = jax.random.PRNGKey(0)
    n_samples = len(z_samples)

    key,subkey= jax.random.split(key)
    z_samples_eval = jax.random.normal(subkey,(n_samples,2))
    z_samples_eval = jax.device_put(z_samples_eval,device)
    # Perform gradient flow step
    current_node, step_info = gradient_flow_step(
        current_node, z_samples_eval, G_mat, potential,
        step_size=h, 
        solver_tol=tolerance,
        regularization=regularization
    )
    
    # Evaluate new energy
    _, current_params = nnx.split(current_node)
    current_energy, samples0 = potential.evaluate_energy(current_node, z_samples, current_params)
    
    # Store diagnostics
    energy_history.append(float(current_energy))
    solver_stats.append(step_info)
    param_norm = jnp.sqrt(sum(jax.tree.leaves(jax.tree.map(lambda x: jnp.sum(x**2), current_params))))
    param_norms.append(float(param_norm))
    
    # Store samples for visualization at regular intervals
    sample_history.append(samples0)
    
    for iteration in range(max_iterations):
        
        key,subkey= jax.random.split(key)
        z_samples_eval = jax.random.normal(subkey,(n_samples,2))
        z_samples_eval = jax.device_put(z_samples_eval,device)
        # Perform gradient flow step
        current_node, step_info = gradient_flow_step(
            current_node, z_samples_eval, G_mat, potential,
            step_size=h, 
            solver=solver,
            solver_tol=tolerance,
            regularization=regularization
        )
        
        # Evaluate new energy
        _, current_params = nnx.split(current_node)
        current_energy, samples1 = potential.evaluate_energy(current_node, z_samples, current_params)
        
        # Store diagnostics
        energy_history.append(float(current_energy))
        solver_stats.append(step_info)
        param_norm = jnp.sqrt(sum(jax.tree.leaves(jax.tree.map(lambda x: jnp.sum(x**2), current_params))))
        param_norms.append(float(param_norm))
        
        # Store samples for visualization at regular intervals
        sample_history.append(samples1)
        
        # Progress reporting
        if iteration % progress_every == 0 :
            energy_decrease = energy_history[0] - current_energy
            print(f"Iter {iteration:3d}: Energy = {current_energy:.6f}, "
                  f"Decrease = {energy_decrease:.6f}, "
                  f"Grad norm: {step_info['gradient_norm']:.2e}")
            fig,ax = plt.subplots()
            ax = potential.plot_function(fig = fig, ax=ax)
            ax.scatter(samples0[:,0],samples0[:,1],color = 'blue',s=5,alpha = 0.1,label = f'Iteration {iteration-1}')
            ax.scatter(samples1[:,0],samples1[:,1],color = 'red',s=5,alpha = 0.1,label = f'Iteration {iteration}')
            ax.set_title(f"Iteration {iteration}: Energy = {current_energy:.6f}")
            plt.legend()
            plt.show()
            
            plt.close(fig)
            # Update previous samples
            samples0 = samples1
        # Early stopping conditions
        if current_energy < tolerance:
            print(f"Converged! Energy below tolerance at iteration {iteration}")
            break
            
                    
        if iteration > 5 and abs(energy_history[-1] - energy_history[-2]) < tolerance * 1e-2:
            print(f"Energy increment below tolerance at {iteration}")
            break
    
    # Final summary
    final_energy = energy_history[-1]
    total_decrease = energy_history[0] - final_energy
    
    print(f"\n=== Integration Complete ===")
    print(f"Total iterations:    {len(energy_history)-1}")
    print(f"Initial energy:      {energy_history[0]:.6f}")
    print(f"Final energy:        {final_energy:.6f}")
    print(f"Total decrease:      {total_decrease:.6f}")
    print(f"Reduction ratio:     {final_energy/energy_history[0]:.4f}")
    print(f"Final param norm:    {param_norms[-1]:.6f}")
    
    return {
        'final_node': current_node,
        'energy_history': energy_history,
        'param_norms': param_norms,
        'sample_history': sample_history,
        'potential': potential,
        'convergence_info': {
            'converged': final_energy < tolerance or abs(energy_history[-1] - energy_history[-2]) < tolerance * 1e-2,
            'final_energy': final_energy,
            'total_decrease': total_decrease,
            'iterations': len(energy_history) - 1
        }
    }
