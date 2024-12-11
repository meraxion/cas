import os
import matplotlib.pyplot as plt
import numpy as np
import scipy
from typing import Callable

from vindy.utils import add_lognormal_noise

### DYNAMICAL SYSTEM MODELS ###
# Implements a (Lorenz attractor) solver
def get_euler(f:Callable, dt:float):
  def euler_scan_fn(carry, t):
    y, args = carry

    y_next = y + dt * f(t, y, args)

    carry = y_next, args
    return carry, y_next
  return euler_scan_fn

def lorenz_scan(t, y, params):
  """jax-compatible """
  
  sigma, rho, beta = params

  x1, x2, x3 = y

  dx1dt = sigma*(x2 - x1)
  dx2dt = x1*(rho - x3) - x2
  dx3dt = x1*x2 - beta*x3

  y_next = dx1dt, dx2dt, dx3dt

  return y_next

def lorenz(t, x, params:tuple[float, float, float]=(10,28,8/3)):
  x1, x2, x3 = x
  sigma, rho, beta = params

  dx1dt = sigma*(x2 - x1)
  dx2dt = x1*(rho - x3) - x2
  dx3dt = x1*x2 - beta*x3

  y_next = dx1dt, dx2dt, dx3dt

  return y_next

def lkv(t, x, params:tuple[float, float, float, float]=(1.1,0.4,0.1,0.4)):
  x1, x2 = x
  alpha, beta, delta, gamma = params

  dx1dt = (alpha - beta*x2)*x1
  dx2dt = (delta*x1 - gamma)*x2

  y_next = dx1dt, dx2dt

  return y_next

### VARIOUS UTILS ### 
def gen_dirs(model_name, sindy_type, scenario_info, outdir):
  # 
  outdir = os.path.join(outdir, f"{model_name}", f"{sindy_type}")
  figdir = os.path.join(outdir, "figures", f"{scenario_info}")
  log_dir = os.path.join(outdir, f"{model_name}", "log", f"{scenario_info}")
  weights_dir = os.path.join(outdir, "weights", f"{scenario_info}")

  # save fig 
  for dir in [outdir, figdir, log_dir, weights_dir]:
    if not os.path.isdir(dir):
      os.makedirs(dir)

  return outdir, figdir, log_dir, weights_dir

def gen_ics(seed, n_train, n_test, ic, mdl_params, mdl_noise):
  """
  generates noisy initial conditions and coefficients
  """
  np.random.seed(seed)

  x0 = np.concatenate(
    [np.random.normal(ic_, scale=2, size=(n_train + n_test, 1)) for ic_ in ic],
    axis = 1)
  
  params = []
  np.random.seed(seed)
  for i in range(mdl_params):
    params.append(np.random.normal(mdl_params[i], mdl_params[i]*mdl_noise,
                                   size=n_train))
    
  return x0, np.array(params)

def gen_data(f:Callable, x0, ts, params, n_train):
  """
  generates data and adds measurement noise
  """
  x = np.array(
    [
      scipy.integrate.odeint(lambda x_, t: f(t, x_, params[i]), x0_, ts) 
      for i, x0_ in enumerate(x0[:n_train])
    ]
  )

  x = np.array([add_lognormal_noise(x_, )])