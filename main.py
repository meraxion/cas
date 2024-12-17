# Generic imports
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
# tools import
from typing import Callable
from itertools import product
# vindy import
from vindy import SindyNetwork
from vindy.libraries import PolynomialLibrary, BaseLibrary
from vindy.layers import SindyLayer, VindyLayer
from vindy.distributions import Gaussian, Laplace
from vindy.callbacks import (
  SaveCoefficientsCallback,
)
# local utils import
from proj_utils import lorenz, gen_dirs, gen_ics, gen_data, get_time_derivatives, plot_lorenz

# This first section mostly follows the example provided in
# https://colab.research.google.com/drive/1Tvk93iU5kh7i7ffkOwfMUPwxT1rhhoW0


def train_vindy_model(
    sindy_type:str="vindy",
    model_name:str="lorenz",
    dynamics:Callable=lorenz,
    dynamics_params:np.ndarray=np.array([10, 28, 8/3]),
    seed:int=37,
    mdl_noise:float=0,
    measurement_noise:float=0,
    mdl_library:list[BaseLibrary]=[PolynomialLibrary(3, 3)],
    n_train:int=30,
    n_test:int=4,
    learning_rate:float=0.001,
    beta:float=1e-3,
    l_dz:float=1e0,
    epochs:int=500,
    batch_size:int=256,
    pdf_threshold:float=0.5,
    prior_distribution=Laplace(0.0, 1.0)):
    """
    train a VINDy model with given hyperparameters.
    
    args
    sindy_type: Type of SINDy model. takes vindy or sindy
    model_name: Name of the model
    seed: Random seed
    mdl_noise: Model noise level
    measurement_noise: Measurement noise level
    n_train: Number of training trajectories
    n_test: Number of test trajectories
    learning_rate: Learning rate for optimizer
    beta: Beta parameter for VINDy layer
    l_dz: l_dz parameter for SINDy network
    epochs: Number of training epochs
    batch_size: Training batch size
    pdf_threshold: Threshold for PDF thresholding
    prior_distribution: Prior distribution for VINDy layer
    
    Returns:
        dict: Training history and model metrics
    """
    scenario_info = f"{sindy_type}_mdl_noise_{mdl_noise}_seed_{seed}_noise_{measurement_noise}"
    _, _, _, weights_dir = gen_dirs(model_name, sindy_type, scenario_info, "results")
    
    # Initial conditions
    ic = [0, 0, 25]  # [x1_0, x2_0, x3_0]
    
    # Time vector
    t0, T, dt = 0, 25, 0.01
    ts = np.arange(t0, T, dt)
    nt = ts.shape[0]
    
    # Generate data
    x0, params = gen_ics(seed, n_train, n_test, ic, dynamics_params, mdl_noise)
    x, x_test = gen_data(dynamics, x0, ts, params, n_train, measurement_noise, dynamics_params)
    dxdt, dxdt_test = get_time_derivatives(x, x_test, dt)
    
    ### Make VINDy model ###
    # reshape data to fit model
    x_train = np.concatenate(x, axis=0)
    dxdt_train = np.concatenate(dxdt, axis=0)
    x_test = np.concatenate(x_test, axis=0)
    dxdt_test = np.concatenate(dxdt_test, axis=0)
    
    # create model
    # model param library
    layer_params = dict(
        state_dim=x_train.shape[1],
        param_dim=0,
        feature_libraries=mdl_library,
        second_order=False,
        mask=None,
        kernel_regularizer=tf.keras.regularizers.L1L2(l1=0, l2=0)
    )
    # sindy layer
    sindy_layer = VindyLayer(
        beta=beta,
        priors=prior_distribution,
        **layer_params
    )
    # sindy auto_encoder
    mdl = SindyNetwork(
        sindy_layer=sindy_layer,
        x=x_train,
        l_dz=l_dz,
        dt=dt,
        second_order=False
    )
    
    mdl.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="huber")
    
    input_shapes = [(x_train.shape[1],), (dxdt_train.shape[1],)]
    mdl.build(input_shapes)
    
    weights_path = os.path.join(weights_dir, f"weights_{scenario_info}.h5")

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
    
    # Train model
    trainhist = mdl.fit(
        x=[x_train, dxdt_train],
        callbacks=callbacks,
        y=None,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    
    # Load best weights and apply threshold
    mdl.load_weights(weights_path)
    sindy_layer.pdf_thresholding(threshold=pdf_threshold)
    
    # Calculate test loss
    test_loss = mdl.evaluate([x_test, dxdt_test], verbose=0)
    
    return {
        'history': trainhist.history,
        'test_loss': test_loss,
        'model': mdl,
        'scenario_info': scenario_info
    }

def run_hyperparameter_sweep():
    """
    Run a hyperparameter sweep over different combinations of parameters.
    """
    # Define parameter grid
    param_grid = {
        'measurement_noise': [0, 0.1, 0.2],
        'mdl_noise': [0, 0.05, 0.1],
        'beta': [1e-4, 1e-3, 1e-2],
        'l_dz': [1e-1, 1e0, 1e1],
        'learning_rate': [1e-4, 1e-3],
        'pdf_threshold': [0.3, 0.5, 0.7]
    }
    
    results = []
    
    # Generate all combinations of parameters
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in product(*values)]
    
    for params in param_combinations:
        print(f"\nTraining with parameters: {params}")
        try:
            result = train_vindy_model(**params)
            results.append({
                'params': params,
                'final_loss': result['history']['loss'][-1],
                'test_loss': result['test_loss'],
                'scenario_info': result['scenario_info']
            })
        except Exception as e:
            print(f"Error with parameters {params}: {str(e)}")
    
    return results

def main():
    # set random seeds for reproducibility
    np.random.seed(37)
    tf.random.set_seed(37)
    
    # run hyperparameter sweep
    results = run_hyperparameter_sweep()
    
    # save results
    results_dir = "hyperparameter_sweep_results"
    os.makedirs(results_dir, exist_ok=True)
    np.save(os.path.join(results_dir, "sweep_results.npy"), results)
    
    # print best results
    sorted_results = sorted(results, key=lambda x: x['test_loss'])
    print("\nTop 5 configurations:")
    for i, result in enumerate(sorted_results[:5]):
        print(f"\n{i+1}. Test Loss: {result['test_loss']:.6f}")
        print("Parameters:", result['params'])

def test():
  # Set random seeds for reproducibility
  np.random.seed(37)
  tf.random.set_seed(37)

  # First, let's mostly reproduce a Lorenz attractor under ideal conditions
  # That means, full observability, no noise, large-ish (polynomial library)


  # run test

  return

if __name__ == "__main__":
  # main()
  test()