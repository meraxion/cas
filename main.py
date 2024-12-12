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

from proj_utils import lorenz, gen_dirs, gen_ics, gen_data, get_time_derivatives, plot_lorenz

# First, let's mostly reproduce a Lorenz attractor under ideal conditions
# That means, full observability, no noise, large-ish (polynomial library)
# This first section mostly follows the example provided in
# https://colab.research.google.com/drive/1Tvk93iU5kh7i7ffkOwfMUPwxT1rhhoW0

# General script parameters
sindy_type = "vindy"
model_name = "lorenz"
seed = 37

mdl_noise = 0
measurement_noise = 0

n_train = 30 # n train trajectories
n_test  = 4  # n test  trajectories

scenario_info = f"{sindy_type}_mdl_noise_{mdl_noise}_seed_{seed}_noise_{measurement_noise}"
_, _, _, weights_dir = gen_dirs(model_name, sindy_type, scenario_info, "results")

# initial conditions
x1_0 = 0
x2_0 = 0
x3_0 = 25
ic = [x1_0, x2_0, x3_0]
dim = 3
var_names = ["x_1", "x_2", "x_3"]

# time vector
t0, T, dt = 0, 25, 0.01
ts = np.arange(t0, T, dt)
nt = ts.shape[0]

mdl_params = np.array([10, 28, 8/3])

x0, params = gen_ics(seed, n_train, n_test, ic, mdl_params, mdl_noise)

x, x_test = gen_data(lorenz, x0, ts, params, n_train, measurement_noise,mdl_params)

dxdt, dxdt_test = get_time_derivatives(x, x_test, dt)

plot_lorenz(x, x_test)

### Make VINDy model ###
# reshape data to fit model
x_train = np.concatenate(x, axis=0)
dxdt_train = np.concatenate(dxdt, axis=0)
x_test = np.concatenate(x_test, axis=0)
dxdt_test = np.concatenate(dxdt_test, axis=0)

# model parameters 
libraries = [
  PolynomialLibrary(3, 3)
]

# create sindy layer
layer_params = dict(
  state_dim = x_train.shape[1],
  param_dim = 0,
  feature_libraries = libraries,
  second_order = False,
  mask = None,
  kernel_regularizer = tf.keras.regularizers.L1L2(l1=0, l2=0)
)
sindy_layer = VindyLayer(
  beta=1e-3,
  priors=Laplace(0.0, 1.0),
  **layer_params
)

# create autoencoder sindy model
mdl = SindyNetwork(
  sindy_layer=sindy_layer,
  x=x_train,
  l_dz = 1e0,
  dt=dt,
  second_order=False
)

mdl.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="huber")
input_shapes = [(x_train.shape[1],), (dxdt_train.shape[1],)]
mdl.build(input_shapes)

weights_path = os.path.join(weights_dir, ".weights.h5")

callbacks = [
  tf.keras.callbacks.ModelCheckpoint(
    filepath = os.path.join(weights_path),
    save_weights_only = True,
    save_best_only = True,
    monitor = "loss",
    verbose = 0
  ),
  SaveCoefficientsCallback()
]

trainhist = mdl.fit(
  x = [x_train, dxdt_train],
  callbacks = callbacks,
  y=None,
  epochs = 500,
  batch_size = 256,
  verbose = 1
)

# load best weights
mdl.load_weights(os.path.join(weights_path))
# apply pdf threshold
threshold = 0.5
sindy_layer.pdf_thresholding(threshold=threshold)

