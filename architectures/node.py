import os
import sys
# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import jax.numpy as jnp
from typing import  Optional,Tuple
from jaxtyping import PyTree,Array
from ODE_solvers.solvers import  string_2_solver
from flax import nnx
import jax

# Neural ODE class
class NeuralODE(nnx.Module):
    def __init__(self, 
                 dynamics_model = nnx.Module,
                 time_dependent: bool = False,
                 solver: str = "euler",
                 dt0=0.1,
                 rtol=1e-4,
                 atol=1e-6):
        self.dynamics = dynamics_model
        self.solver = string_2_solver(solver)
        self.dt0 = dt0
        self.rtol = rtol
        self.atol = atol
        self.time_dependent = time_dependent

    # Define the vector field function
    def vector_field(self,t, y, args):
        data = y
        if self.time_dependent:
            data = jnp.concatenate([t[:,None], y], axis=-1)  # Add time as a feature
        return self.dynamics(data)
    
    def log_likelihood(self):
        #TODO
        return
    def score_function(self):
        #TODO
        return

    # @nnx.jit
    def __call__(self, y0: Array, t_span: Optional[Tuple[float,float]] = (0.0,1.0), params: Optional[PyTree] = None) -> Array:
        """
        Solve the ODE from t_span[0] to t_span[1] with initial condition y0
        
        Args:
            y0: Initial condition, shape (batch_size, feature_dim) or (feature_dim,)
            t_span: Tuple of (t0, t1) for integration bounds
            
        Returns:
            Final state at time t1
        """
        
        if params is None:
            model = self.dynamics
        else:
            graphdef,_ = nnx.split(self.dynamics)
            model = nnx.merge(graphdef, params)
        # Use defined model for the vector field
        def vector_field(t: float, y: Array, args: Optional[dict] = None):
            data = y
            if self.time_dependent:
                t = t*jnp.ones((y.shape[0],))  # Broadcast time to match batch size
                data = jnp.concatenate([t[:,None], y], axis=-1)  # Add time as a feature
            return model(data)
        
        t_list = jnp.arange(t_span[0], t_span[1], self.dt0)

        y = self.solver(vector_field,t_list,y0,history=False)
       
        
        return y.reshape(-1,y0.shape[-1])  # Return final state

