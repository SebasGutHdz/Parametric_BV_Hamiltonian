import jax
import jax.numpy as jnp
from typing import Callable, Dict, Any, Optional


def euler_method(f,t_list, y0, history=False):
    """
    Euler method for solving ODEs.
    Inputs:
        f: Callable , the function defining the ODE dy/dt = f(t, y), where t is a float and y is a tensor of shape [bs, d]
        t_list: jax array, a list of time points at which to evaluate the ODE 
        y0: tensor [bs,d], initial value of y at t0
    Outputs:
    """
    n_steps = len(t_list) - 1
    if history:
        y_h = [y0]
        ys = y_h[-1]
    else:
        ys = y0
    for i in range(n_steps):
        dt = t_list[i + 1] - t_list[i]
        ts = t_list[i]
        y_next = ys + dt * f(ts, ys) # Euler step [Bs, d]
        if history:
            y_h.append(y_next)
            ys = y_h[-1]
        else:
            ys = y_next
    if history:
        return jnp.array(y_h)
    else:
        return ys

def heun_method(f, t_list, y0, history=False):
    """
    Heun's method for solving ODEs.
    Inputs:
        f: Callable , the function defining the ODE dy/dt = f(t, y), where t is a float and y is a tensor of shape [bs, d]
        t_list: jax array, a list of time points at which to evaluate the ODE 
        y0: tensor [bs,d], initial value of y at t0
    Outputs:
        y_h: jax array, the solution of the ODE at the time points in t_list
    """
    n_steps = len(t_list) - 1
    if history:
        y_h = [y0]
        ys = y_h[-1]
    else:
        ys = y0
    for i in range(n_steps):    
        dt = t_list[i+1] - t_list[i]
        ts = t_list[i]
        y_mid = ys + dt*f(ts,ys) # Euler step [Bs, d]
        y_next = ys + dt/2*(f(ts,ys)+f(t_list[i+1],y_mid)) # Heun step [Bs, d]
        if history:
            y_h.append(y_next)
            ys = y_h[-1]
        else:
            ys = y_next
    if history:
        return jnp.array(y_h)
    else:
        return ys
    
def string_2_solver(solver_str: str):
    """
    Converts a string to a diffrax solver.
    Inputs:
        solver_str: str, the name of the solver
    Outputs:
        solver: diffrax solver, the corresponding diffrax solver
    """
    if solver_str == 'euler':
        return euler_method
    elif solver_str == 'heun':
        return heun_method
    else:
        raise ValueError(f'Solver {solver_str} not recognized. Available solvers: euler, heun.')
    # if solver_str == 'euler':
    #     return diffrax.Euler()
    # elif solver_str == 'heun':
    #     return diffrax.Heun()
    # else:
    #     raise ValueError(f'Solver {solver_str} not recognized. Available solvers: euler, heun.')
