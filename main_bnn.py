import torch
import pyro
import numpy as np
import argparse

from pyro.infer import Predictive
from visualize import visualize_prediction
from utils_read import data_mkdir, de_normalize
from tqdm.auto import trange
from evaluate import eval_all
from models.bnn_models import BNN
from data_loader import create_dataset

mcmc_mlp_path='results/mcmc-mlp'
data_mkdir(mcmc_mlp_path)

svi_mlp_path='results/svi-mlp'
data_mkdir(svi_mlp_path)


def svi(model, X_train, y_train, X_test, num_epochs, lr=0.01):
    from pyro.infer import SVI, Trace_ELBO
    from pyro.infer.autoguide import AutoDiagonalNormal
    from tqdm.auto import trange
    
    pyro.clear_param_store()
    mean_field_guide = AutoDiagonalNormal(model)
    optimizer = pyro.optim.Adam({"lr": lr})

    svi = SVI(model, mean_field_guide, optimizer, loss=Trace_ELBO())
    pyro.clear_param_store()

    progress_bar = trange(num_epochs)

    for _ in progress_bar:
        loss = svi.step(X_train, y_train)
        progress_bar.set_postfix(loss=f"{loss / X_train.shape[0]:.3f}")
        
    predictive = Predictive(model, guide=mean_field_guide, num_samples=500)
    preds = predictive(X_test)
    preds_npy = preds['obs'].T.detach().numpy()
    np.save(f'{svi_mlp_path}/svi_pred.npy', preds_npy)
    return preds_npy


def mcmc(model, X_train, y_train, X_test, num_samples=41*2):
    from pyro.infer import MCMC, NUTS
    nuts_kernel = NUTS(model, jit_compile=True)
    mcmc = MCMC(nuts_kernel, num_samples=num_samples)
    mcmc.run(X_train, y_train)

    predictive = Predictive(model=model, posterior_samples=mcmc.get_samples())
    preds = predictive(X_test)
    preds_npy = preds['obs'].T.detach().numpy()
    np.save(f'{mcmc_mlp_path}/mcmc_pred.npy', preds_npy)
    return preds_npy    
    
    
if __name__ == '__main__':
    method = "mcmc"
    X_train, X_test, y_train, y_test = create_dataset()
    
    model = BNN(in_dim=7, out_dim=1, hid_dim=96, n_hid_layers=2, prior_scale=5.)
    
    if method == "svi":
        # preds_npy = svi(model, X_train, y_train, X_test, num_epochs=25000, lr=0.01)
        preds_npy = np.load(f'{svi_mlp_path}/svi_pred.npy')
        save_file=f"{svi_mlp_path}/svi_mlp.log"
    else:
        # preds_npy = mcmc(model, X_train, y_train, X_test, num_samples=100)
        reds_npy = np.load(f'{mcmc_mlp_path}/mcmc_pred.npy')
        save_file=f"{mcmc_mlp_path}/mcmc_mlp.log"
        
    y_test, y_pred = de_normalize(y_test,  preds_npy)
    y_pred[y_pred < 0] = 0
    
    y_mean = y_pred.mean(axis=1)
    y_std = y_pred.std(axis=1)
    
    eval_all(y_test, y_mean, save_file=save_file)
    visualize_prediction(
        y_test, y_mean, y_pred_std=y_std, 
        img_path=svi_mlp_path, interval=1)