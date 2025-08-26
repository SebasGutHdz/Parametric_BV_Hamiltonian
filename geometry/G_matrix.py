import os
import sys
# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
from jax import random as jrandom
from jax import jit, vmap,grad, flatten_util
from typing import Dict, Any,Optional
from jaxtyping import PyTree,Array
from functools import partial
from jax.scipy.sparse.linalg import cg,gmres
from flax import nnx

from architectures.node import Parametric_NODE
from geometry.lin_alg_solvers import minres

class G_matrix:
    '''
    Computation of G matrix
    '''

    def __init__(self, node: nnx.Module ):
        
        '''
        Initialize G matrix computation 

        Args:
            node: Neural ODE model nnx.Module instance     
        '''

        self.node = node
        

    @partial(jit,static_argnums = (0,))
    def mvp(self,z_samples: Array, eta: PyTree, params: Optional[PyTree] = None)-> PyTree:
        '''
        Computation of G eta
        Args:
            z_samples: (Bs,d) Samples from reference density
            eta: PyTree with same GraphDef as node
            parms: PyTree where the G matrix is computed at
        Return:
            G(theta) eta : PyTree
        '''

        if params is None:
            
            _,params = nnx.split(self.node)
        
        def single_sample_contribution(z: Array)-> PyTree:

            # Define the flow map

            def flow_map(p):

                return self.node(z.reshape(1,-1), (0.0, 1.0), params=p)

            # Step 1: Compute \partial_{theta}T @ eta using Jvp

            jvp_result = jax.jvp(flow_map,(params,),(eta,))[1]

            # Step 2: Compute \partial_{\theta}T @ jvp_result

            _,vjp_fn = jax.vjp(flow_map,params)

            result = vjp_fn(jvp_result)[0]

            return result
    
        # Vectorize over all samples

        contributions = vmap(single_sample_contribution)(z_samples)

        return jax.tree.map(lambda x: jnp.mean(x, axis=0), contributions)
    
    # @partial(jit,static_argnums = (0,6))
    def solve_system(self,z_samples: Array, b: PyTree, params: Optional[PyTree] = None, tol: float = 1e-5, maxiter: int = 10, method: str = "cg", x0: Optional[PyTree] = None) -> PyTree:

        '''
        Solve G(theta) x = b using conjugate gradient method

        Args:
            z_samples: (Bs,d) Samples from reference density
            b: PyTree with same GraphDef as node
            parms: PyTree where the G matrix is computed at
            tol: Tolerance for CG solver
            maxiter: Maximum number of iterations for CG solver
            method: Method to use for solving the linear system ("cg" or "gmres")
            x0: Initial guess for the solution

        Returns:
            x: PyTree solution to G(theta)x = b
        '''
        if method not in ["cg","gmres","minres"]:
            raise ValueError(f"Unknown method: {method}")
        if method == "cg":
            solver = cg
        elif method == "gmres":
            solver = gmres
        elif method == "minres":
            solver = minres
        if params is None:
            _,params = nnx.split(self.node)
        # Define the linear operator for G(theta)
        matvec = lambda eta: self.mvp(z_samples, eta, params)
        # Use Jax inbuilts methods cg or gmres. 
        x,info = solver(matvec,b,tol = tol, maxiter = maxiter,x0=x0)
        
        # x,info = minres(matvec, b, tol=tol, maxiter=maxiter,x0 = x0)
        return x


@partial(jax.jit, static_argnums=(0))  # Static node
def G_matvec_vmap_optimized(node, trainable_params, fixed_params, v_trainable, 
                           z_samples, time):
    """
    Input:
    - node: Parametric_NODE instance
    - full_params: Dictionary containing 'params' and 'fixed' parameters
    - v_trainable: Tangent vector for the trainable parameters
    - z_samples: Samples from reference distribution for contributions
    - device: JAX device to run the computation on
    - t_final: Final time for the flow map
    - n_time_steps: Number of time steps for the flow map
    Output:
    - Gv: G matrix-vector product, approximated using monte carlo sampling
    """
    
    def single_sample_contrib(z):
        """Compute contribution for a single sample z
        Input:
        - z: Single sample from the reference distribution
        Output:
        - Jvp: Jacobian-vector product
        """
        def flow_map(params):
            full_params = {'params': params, 'fixed': fixed_params}
            return node.forward(parameters=full_params, t_list=time, 
                               y0=z.reshape(1, -1), history=False)
        
        # Compute jvp, signature (function, variable, tangent vector) , returns [evaluation, jvp]
        u = jax.jvp(flow_map, (trainable_params,), (v_trainable,))[1] 
        # Compute vjp, signature (function, variable (unpacked)), returns [evaluation, function to evaluate vjp]
        return jax.vjp(flow_map, trainable_params)[1](u)[0] # obtain output of vjp
    
    # vmap over samples. If n_samples is large, this will be efficient
    contributions = jax.vmap(single_sample_contrib)(z_samples)

    
    
    # Efficient reduction
    return jax.tree.map(lambda x: jnp.mean(x, axis=0), contributions)

def compute_Gv_vmap(node, full_params, v_trainable, z_samples, device, **kwargs):
    """Interface for vmap version"""
    
    # Pre-compute static data
    time = jnp.linspace(0, kwargs.get('t_final', 1.0), kwargs.get('n_time_steps', 20))
    
    # Single device placement
    data = jax.device_put({
        'trainable': full_params['params'],
        'fixed': full_params['fixed'], 
        'v': v_trainable,
        'z': z_samples,
        'time': time
    }, device)
    
    return G_matvec_vmap_optimized(
        node, data['trainable'], data['fixed'], 
        data['v'], data['z'], data['time']
    )