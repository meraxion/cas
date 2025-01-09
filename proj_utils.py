import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import scipy
from typing import Callable

import scipy.integrate
from vindy.utils import add_lognormal_noise
from vindy.layers import VindyLayer

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
  """jax-compatible"""
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


### USE TRAINED MODEL TO PREDICT INTO THE FUTURE ###
def sindy_predict(mdl, 
                  sindy_layer:VindyLayer, 
                  x_test, 
                  ts, dim, var_names, 
                  fig_dir,
                  n_traj = 10, i_test = 0):
  
  kernel_orig, kernel_scale_orig = sindy_layer.kernel, sindy_layer.kernel_scale

  # integrate basic model
  t_0 = i_test * int(ts.shape[0])

  t_preds = []
  x_preds = []

  nt = ts.shape[0]
  t_0 = i_test * int(nt)

  print(f"test_trajectory {i_test}")
  for traj in range(n_traj):
    print(f"\t sample {traj+1} out of {n_traj}")
    # sample from the posterior distribution of the coefficients
    # and remove zeroes and other NAs
    sampled_coeff, f_, ff_ = sindy_layer._coeffs
    sampled_coeff = sampled_coeff.numpy()
    sampled_coeff = sampled_coeff[sampled_coeff != 0]

    # assign the sampled coefficients to the sindy layer
    sindy_layer.kernel = tf.reshape(sampled_coeff, (-1,1))
    sol = mdl.integrate(x_test[t_0:t_0+1].squeeze(), ts.squeeze())
    t_preds.append(sol.t)
    x_preds.append(sol.y)

  # restore original coefficients
  sindy_layer.kernel, sindy_layer.kernel_scale = kernel_orig, kernel_scale_orig
  # calculate mean and variance of the trajectories
  x_uq = np.array(x_preds)
  x_uq_mean_sampled = np.mean(x_uq, axis=0)
  x_uq_std = np.std(x_uq, axis=0)

  plot_vindy_pred(x_test, ts, nt, dim, var_names, t_preds, i_test,
                  x_uq_mean_sampled, x_uq_std, fig_dir)
  return

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
  for i in range(len(mdl_params)):
    params.append(np.random.normal(mdl_params[i], mdl_params[i]*mdl_noise,
                                   size=n_train))
    
  return x0, np.array(params).T

def gen_data(f:Callable, x0, ts, params, n_train, measurement_noise, mdl_params):
  """
  generates data and adds measurement noise
  """
  x = np.array(
    [
      scipy.integrate.odeint(lambda x_, t: f(t, x_, params[i]), x0_, ts) 
      for i, x0_ in enumerate(x0[:n_train])
    ]
  )
  x = np.array([add_lognormal_noise(x_, measurement_noise)[0] for x_ in x])

  x_test = np.array(
    [
      scipy.integrate.odeint(lambda x_, t: f(t, x_, mdl_params), x0_, ts)
      for x0_ in x0[n_train:]
    ]
  )

  return x, x_test

def get_time_derivatives(x, x_test, dt):
  dxdt = [np.array(np.gradient(x_, dt, axis=0)) for x_ in x]
  dxdt_test = [np.array(np.gradient(x_, dt, axis=0)) for x_ in x_test]

  return dxdt, dxdt_test

### PLOTTING UTILS ###
def plot_lorenz(x, x_test, plots_dir):
  fig = plt.figure()

  ax = fig.add_subplot(projection = "3d")
  for i, x_ in enumerate(x):
    if i == 0:
      ax.plot(x_[:,0], x_[:, 1], x_[:, 2], c="gray", label = "Training data")
    else:
      ax.plot(x_[:,0], x_[:, 1], x_[:, 2], c="gray")
  for i, x_ in enumerate(x_test):
    if i == 0:
      ax.plot(x_[:,0], x_[:, 1], x_[:, 2], c="red", label = "Test data")
    else:
      ax.plot(x_[:,0], x_[:, 1], x_[:, 2], c="red")

  plt.xlabel("$x_1$")
  plt.ylabel("$x_2$")
  ax.set_zlabel("$x_3$")
  plt.legend()
  plt.tight_layout()
  plt.show()

  plt.savefig(os.path.join(plots_dir, "_dynamics.png"))
  plt.close()

  return 

def plot_train_hist(history,
                    sindy_layer:VindyLayer,
                    var_names,
                    plots_dir):
  
  plt.figure()
  plt.title("Loss over epochs")
  plt.semilogy(history["loss"])
  plt.semilogy(history["dz"])
  plt.semilogy(history["kl_sindy"])
  plt.legend(["total loss", "dz", "kl_sindy"])
  plt.xlabel("Epochs")
  plt.ylabel("Loss")
  plt.show()

  plt.savefig(os.path.join(plots_dir, "_train_loss.png"))
  plt.close()

  plt.figure()
  plt.title("VINDy coefficients over epochs")
  plt.plot(np.array(history["coeffs_mean"]).squeeze())
  plt.legend()
  plt.xlabel("Epoch")
  plt.ylabel("Coefficient")
  
  equation = sindy_layer.model_equation_to_str(z=var_names, precision=3)
  sindy_layer.visualize_coefficients(x_range = [-1.6, 1.6], z=var_names, mu=None)
  plt.suptitle(equation)
  plt.tight_layout()
  plt.savefig(os.path.join(plots_dir, "_coefficients.png"))
  plt.show()
  plt.close()

  return

def plot_vindy_pred(x_test, ts, nt, dim, var_names, t_preds, i_test, x_uq_mean_sampled, x_uq_std, fig_dir):
  """
  predicts the lorenz attractor forward in time, sampling trajectories from the model parameters to include some uncertainty-quantification in the prediction
  """

  t_0 = i_test * int(nt)

  # UQ plot
  fig, axs = plt.subplots(dim, 1, figsize=(10, 6), sharex=True)
  fig.suptitle(f"Integrated Test Trajectories")
  t_0 = i_test * int(nt)
  axs[0].set_title(f"Test Trajectory {i_test + 1}")

  for i in range(dim):
      axs[i].fill_between(
          t_preds[i],
          x_uq_mean_sampled[i] - 3 * x_uq_std[i],
          x_uq_mean_sampled[i] + 3 * x_uq_std[i],
          color="grey",
          alpha=0.3,
          label="UQ bounds ($\pm 3$ std)"
      )
      axs[i].plot(ts, x_test[t_0: t_0 + nt, i], color="black", label=f"${var_names[i]}$ true")
      axs[i].plot(t_preds[i], x_uq_mean_sampled[i], color="orange", linestyle="--", label=f"${var_names[i]}$ pred mean")
      # Adjust the legend to be outside the plot
      axs[i].legend(loc='upper left', bbox_to_anchor=(1, 1))

  plt.tight_layout(rect=[0, 0, 0.8, 1])  # Adjust the layout to make space for the legends
  plt.savefig(os.path.join(fig_dir, "_predictions.png"))

  plt.show()

  return

def save_train_plots(
    x:list,
    x_test:list,
    history:dict,
    sindy_layer: VindyLayer,
    var_names:list,
    fig_dir:str,
    dynamics_plot:Callable=plot_lorenz
):
  dynamics_plot(x, x_test, fig_dir)

  plot_train_hist(history,
                  sindy_layer,
                  var_names,
                  fig_dir)  
  return

def plot_loss_over_noise(results, param_grid, plots_dir="figures"):

  noises = param_grid["measurement_noise"]
  df = pd.DataFrame.from_dict(results)
  df = df[["final_loss", "test_loss"]]

  plt.figure()
  plt.title("Loss at different levels of measurement noise")
  plt.xlabel("Measurement Noise")
  plt.ylabel("Loss")

  plt.plot(noises, df["final_loss"], label="Final Training Loss")
  plt.plot(noises, df["test_loss"], label="Test Loss")
  plt.legend()
  plt.show()
  save_dir = os.path.join("results", plots_dir)
  plt.savefig(os.path.join(save_dir, "loss_by_noise.png"))
  plt.close()

  return