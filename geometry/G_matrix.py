import os
import sys
# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
from jax import random as jrandom
from jax import jit, vmap
from typing import Dict, Any
from architectures.node import Parametric_NODE
from jax import flatten_util
from functools import partial



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