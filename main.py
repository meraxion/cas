import tensorflow as tf
import os
import scipy
import numpy as np
import matplotlib.pyplot as plt

from vindy import SindyNetwork
from vindy.libraries import PolynomialLibrary
from vindy.layers import SindyLayer, VindyLayer
from vindy.distributions import Gaussian, Laplace
from vindy.callbacks import (
  SaveCoefficientsCallback,
)
from vindy.utils import add_lognormal_noise

from proj_utils import lorenz, gen_dirs

# First, let's mostly reproduce a Lorenz attractor under ideal conditions
# That means, full observability, no noise, large-ish (polynomial library)
# This first section mostly follows the example provided in
# https://colab.research.google.com/drive/1Tvk93iU5kh7i7ffkOwfMUPwxT1rhhoW0

# General script parameters
sindy_type = "vindy"
model_name = "lorenz"
seed = 37

model_noise_factor = 0
measurement_noise_factor = 0

n_train = 30 # n train trajectories
n_test  = 4  # n test  trajectories

scenario_info = f"{sindy_type}_mdl_noise_{model_noise_factor}_seed_{seed}_noise_{measurement_noise_factor}"
_, _, _, weights_dir = gen_dirs(model_name, sindy_type, scenario_info, "results")

# initial conditions
x1_0 = 0
x2_0 = 0
x3_0 = 25
ic = [x1_0, x2_0, x3_0]

# time vector
t0, T, dt = 0, 25, 0.01
ts = np.arange(t0, T, dt)
nt = ts.shape[0]

params = np.array([10, 28, 8/3])
y_out = lorenz(0, ic, params)
