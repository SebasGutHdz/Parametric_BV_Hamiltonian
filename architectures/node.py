import os
import sys
# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import flax.linen as nn
import jax.numpy as jnp
import jax.random as jrandom
from typing import Callable, Dict, Any, Optional, Sequence
from architectures.MMNN import MMNN
from NODEs.solvers import euler_method, heun_method, string_2_solver




class Parametric_NODE():
    '''
    A NODE model using the MMMNN architecture. The ODE solvers are 'euler' and 'heun'.
    We assume that parameters are the output of the module initialization.
    '''

    def __init__(self, 
                 model: nn.Module,
                 parameters: Dict[str, Any],
                 include_time: bool = True,
                 solver: str = 'euler',
                 dt: float = 0.01,
                 history: bool = False):
        self.mmnn = model # type: nn.Module
        if parameters is None:
            raise ValueError("Parameters must be provided for the model.")
        self.current_parameters = parameters # type: Dict[str, Any]
        self.dt = dt # type: float
        self.history = history # type: bool
        self.include_time = include_time # type: bool
        self.solver = string_2_solver(solver) # type: Callable
        
    # Function to obtain init parameters 
    def init_parameters(self, key:jax.random.PRNGKey,y0:jnp.ndarray=None) -> Dict[str, Any]:
        '''
        Initializes the parameters of the model.
        Inputs:
            key: jax.random.PRNGKey, random key for initialization
            y0: jnp.ndarray [bs,d], initial value of y at t0
        Outputs:
            parameters: Dict[str, Any], initialized parameters of the model
        '''
        if y0 is None:
            raise ValueError("Initial state y0 must be provided for initialization.")
        return self.mmnn.init(key, y0)
    #Closure for rhs 
    def closure_rhs(self,parameters:None) -> Callable:
        '''
        Returns a function that computes the right-hand side of the ODE.
        '''
        if parameters is None:
            parameters = self.current_parameters
        
        lambda_rhs = lambda x: self.mmnn.apply(parameters,x.reshape(1,-1)).flatten()

        if self.include_time:
            @jax.jit
            def rhs(t, y):
                '''
                Right-hand side of the ODE.
                Inputs:
                    t: float [], time
                    y: jnp.ndarray [bs,d], state at time t
                Outputs:
                    jnp.ndarray [bs,d], derivative of y at time t
                '''
                t_expand = jnp.full((y.shape[0], 1), t)  # Expand t to match batch size
                ty = jnp.concatenate([t_expand, y], axis=-1)  # Concatenate t and y
                
                return lambda_rhs(ty)
        else:
            @jax.jit
            def rhs(t,y):
                '''
                Right-hand side of the ODE.
                Inputs:
                    t: float, time
                    y: jnp.ndarray, state at time t
                Outputs:
                    jnp.ndarray, derivative of y at time t
                '''
                return lambda_rhs(y)
        
        return rhs

    def get_parameters(self,parameters: Optional[Dict[str,Any]] = None) -> Dict[str, Any]:
        '''
        Returns the current parameters of the model.
        Inputs:
            parameters: Dict[str, Any], parameters for the model
        Outputs:
            parameters: Dict[str, Any], current parameters of the model
        '''
        if parameters is None:
            return self.current_parameters
        else:
            return parameters
        
    def update_parameters(self, parameters: Dict[str, Any]) -> None:
        '''
        Updates the current parameters of the model.
        Inputs:
            parameters: Dict[str, Any], new parameters for the model
        '''
        if not isinstance(parameters, dict):
            raise ValueError("Parameters must be a dictionary.")
        self.current_parameters = parameters

    def forward(self, parameters: Optional[Dict[str, Any]] = None, 
                t_list: Sequence[float] = None,                
                y0: jnp.ndarray = None,
                history: bool = None) -> jnp.ndarray:
        '''
        Forward pass of the NODE model.
        Inputs:
            parameters: Dict[str, Any], parameters for the forward pass
            t_span: Sequence[float], time span for the ODE solver
            y0: jnp.ndarray [bs,d], initial value of y at t0
        outputs:
            y: jnp.ndarray [bs,d], state at each time step in t_span
            
        '''
        parameters = self.get_parameters(parameters)
        if t_list is None:
            t_list = jnp.arange(0, 1, self.dt)
        if y0 is None:
            raise ValueError("Initial state y0 must be provided for the forward pass.")
        if history is None:
            history = self.history

        rhs = self.closure_rhs(parameters)
        y = self.solver(rhs, t_list, y0, history=history)
        
        return y
    
    def eval_vf(self, parameters: Optional[Dict[str, Any]] = None,
                   t: float = 0.0, y: jnp.ndarray = None) -> jnp.ndarray:
        '''
        Evaluates the vector field at a given time and state.
        Inputs:
            parameters: Dict[str, Any], parameters for the forward pass
            t: float [], time at which to evaluate the vector field
            y: jnp.ndarray [bs,d], state at time t
        Outputs:
            jnp.ndarray [bs,d], vector field at time t and y
        ''' 
        parameters = self.get_parameters(parameters)
        if y is None:
            raise ValueError("State y must be provided for evaluating the vector field.")
        rhs = self.closure_rhs(parameters)
        
        return rhs(t,y)
        
    def log_likelihood():
        '''
        Computes the log likelihood of the model given the parameters and data.
        This is a placeholder function and should be implemented in subclasses.
        '''
        raise NotImplementedError("log_likelihood method not implemented.")
    
    def fisher_information():
        '''
        Computes $$\int ||\nabla \rho(x)||^2\rho(x)dx$$.
        This is a placeholder function and should be implemented in subclasses.
        '''
        raise NotImplementedError("fisher_information method not implemented.")
    

def init_node(mmnn: nn.Module, 
              d_space: int = 2, 
              include_time: bool = True) -> Dict[str, Any]:
    '''
    Initializes the parameters of the MMNN model.
    Inputs:
        mmnn: MMNN, the MMNN model to initialize
        d_space: int, the dimension of the state space
        include_time: bool, whether to include time in the input
    Outputs:        
        params: Dict[str, Any], initialized parameters of the MMNN model
    '''
        # Obtain parameters
    t0 = 0.0
    y0 = jnp.zeros((1,d_space))

    if include_time:
        t0 = jnp.array([t0])
        t0 = jnp.expand_dims(t0, axis=0)  # Reshape to [1, 1] for batch size 1
        ty0 = jnp.concatenate([t0, y0], axis=-1)
        params = mmnn.init(jrandom.PRNGKey(0), ty0)
    else:
        params = mmnn.init(jrandom.PRNGKey(0), y0.reshape(1,-1))

    return params