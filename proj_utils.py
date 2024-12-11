import os
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array
from typing import Callable
from jax.random import PRNGKey
from tqdm import tqdm

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

def lorenz(t, x, sigma=10, rho=28, beta=8/3):
  x1, x2, x3 = x

  dx1dt = sigma*(x2 - x1)
  dx2dt = x1*(rho - x3) - x2
  dx3dt = x1*x2 - beta*x3

  y_next = dx1dt, dx2dt, dx3dt

  return y_next

def lkv(t, x, alpha=1.1, beta=0.4, delta=0.1, gamma=0.4):
  x1, x2 = x

  dx1dt = (alpha - beta*x2)*x1
  dx2dt = (delta*x1 - gamma)*x2

  y_next = dx1dt, dx2dt

  return y_next

### VARIOUS UTILS ### 
def generate_directories(model_name, sindy_type, scenario_info, outdir):
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