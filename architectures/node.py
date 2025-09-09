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

from core.types import SampleArray,TimeArray,VelocityArray,TrajectoryArray
from architectures.utils_node import eval_model
from ODE_solvers.log_ODEs import divergence_vf,divergence_vf_hutch,jacobian_vf,compute_jacobian_and_grad_div




# Neural ODE class
class NeuralODE(nnx.Module):
    
    def __init__(self, 
                 dynamics_model: nnx.Module,
                 time_dependent: bool = False,
                 solver: str = "euler",
                 dt0: float = 0.1,
                 rtol=1e-4,
                 atol=1e-6):
        self.dynamics = dynamics_model
        self.solver = string_2_solver(solver)
        self.dt0 = dt0
        self.rtol = rtol
        self.atol = atol
        self.time_dependent = time_dependent

    # Define the vector field function
    def vector_field(self,t: TimeArray, y: SampleArray, args: Optional[dict] = None) -> SampleArray:
        data = y
        if self.time_dependent:
            return eval_model(self.dynamics,t,y)
        return self.dynamics(data)

    def log_likelihood(self,t: TimeArray,xt: TrajectoryArray,log_prob_init: Array,method: str = 'exact',params: Optional[PyTree] = None,log_trajectory: bool = False) -> Array:
        '''
        Solve ODE for loglikelihood of ODE 
        Args:
            t: Time array of shape (time steps,)
            xt: Sample array of shape (batch_size,time steps, dim)
            log_prob_init: Initial loglikelihood at t=0, shape (batch_size,)
            method: Method to compute the loglikelihood, 'exact' or 'hutchinson'
            params: Optional PyTree of parameters for the dynamics model
            log_trajectory: If True, return the full loglikelihood trajectory, else only final value
        Returns:
            log_likelihood: Loglikelihood of the ODE at (t,x), shape (batch_size,)
        '''
        # To call the solver.step method, we need function, t_list, step_index, solution_history
        solution_history = [log_prob_init]
        for i in range(len(t) - 1):
            # Expand t to match batch size
            t_reshape = t[i+1]*jnp.ones(xt[:,i+1,:].shape[0])  # Shape (batch_size,)
            # Implicit rhs of ODE is the negative divergence of the vector field
            log_rhs = lambda time,logp: -self.divergence(t = t_reshape, x=xt[:,i+1,:], method=method, params=params)
            log_new = self.solver.step(log_rhs,t,i,solution_history)
            solution_history.append(log_new)
        if log_trajectory:
            return jnp.array(solution_history)
        else:
            return solution_history[-1]
    def score_function(self,t: TimeArray,xt: TrajectoryArray,score_init: SampleArray,method: str = 'exact',params: Optional[PyTree] = None,score_trajectory: bool = False):
        '''
        Solve ODE for score function of ODE
        Args:
            t: Time array of shape (time steps,)
            xt: Sample array of shape (batch_size,time steps, dim)
            score_init: Initial score at t=0, shape (batch_size, dim)
            method: str: 'exat', 'autodiff'
            params: Optional PyTree of parameters for the dynamics model
            score_trajectory: If True, return the full score trajectory, else only final value
        Returns:
            score: Score function of the ODE at (t,x), shape (batch_size, dim)
        '''
        # To call the solver.step method, we need function, t_list, step_index, solution_history
        solution_history = [score_init]
        for i in range(len(t) - 1):
            # Expand t to match batch size
            t_reshape = t[i]*jnp.ones(xt[:,i,:].shape[0])  # Shape (batch_size,)
            # Negative jacobian of the vector field (bs,dim,dim)
            jacobian,grad_div = self.jacobian_grad_and_div(t = t_reshape, x=xt[:,i,:],method = method, params=params)
            score_rhs = lambda time,score: -jnp.einsum('bij,bj->bi', jacobian, score) - grad_div
            score_new = self.solver.step(score_rhs,t,i,solution_history)
            solution_history.append(score_new)
        if score_trajectory:
            return jnp.array(solution_history)
        else:
            return solution_history[-1]
        

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
                return eval_model(model,t,y)
            return model(data)
        
        t_list = jnp.arange(t_span[0], t_span[1], self.dt0)

        y = self.solver(vector_field,t_list,y0,history=False)
       
        
        return y.reshape(-1,y0.shape[-1])  # Return final state
    
    def divergence(self,t:TimeArray,x:SampleArray,method: str = "exact",params: Optional[PyTree] = None)-> Array:
        if params is None:
            model = self.dynamics
        else:
            graphdef,_ = nnx.split(self.dynamics)
            model = nnx.merge(graphdef, params)
        if method == "exact":
            return divergence_vf(model,t,x)
        elif method == "hutchinson":
            return divergence_vf_hutch(model,t,x)
        
    def jacobian_grad_and_div(self,t:TimeArray,x:SampleArray,method: str = "exact",params: Optional[PyTree] = None)-> Array:
        if params is None:
            model = self.dynamics
        else:
            graphdef,_ = nnx.split(self.dynamics)
            model = nnx.merge(graphdef, params)
        return compute_jacobian_and_grad_div(model,t,x)

