import torch
import pyro
import numpy as np
import argparse

from pyro.infer import Predictive
from visualize import visualize_prediction, save_pred_csv
from utils_read import data_mkdir, de_normalize
from tqdm.auto import trange
from evaluate import eval_all
from models.bnn_models import BNN
from data_loader import create_dataset

mcmc_mlp_path='results/mcmc-mlp'
data_mkdir(mcmc_mlp_path)

svi_mlp_path='results/svi-mlp'
data_mkdir(svi_mlp_path)


def svi(model, X_train, y_train, X_test, args):
    from pyro.infer import SVI, Trace_ELBO
    from pyro.infer.autoguide import AutoDiagonalNormal
    from tqdm.auto import trange
    
    pyro.clear_param_store()
    mean_field_guide = AutoDiagonalNormal(model)
    optimizer = pyro.optim.Adam({"lr":args.lr})

    svi = SVI(model, mean_field_guide, optimizer, loss=Trace_ELBO())
    pyro.clear_param_store()

    progress_bar = trange(args.num_epochs)

    for _ in progress_bar:
        loss = svi.step(X_train, y_train)
        progress_bar.set_postfix(loss=f"{loss / X_train.shape[0]:.3f}")
        
    predictive = Predictive(model, guide=mean_field_guide, num_samples=500)
    preds = predictive(X_test)
    preds_npy = preds['obs'].T.cpu().detach().numpy()
    np.save(f'{svi_mlp_path}/svi_pred.npy', preds_npy)
    return preds_npy


def mcmc(model, X_train, y_train, X_test, args):
    from pyro.infer import MCMC, NUTS
    pyro.clear_param_store()
    nuts_kernel = NUTS(model, jit_compile=True)
    mcmc = MCMC(nuts_kernel, num_samples=args.mcmc_samples)
    mcmc.run(X_train, y_train)

    predictive = Predictive(model=model, posterior_samples=mcmc.get_samples())
    preds = predictive(X_test)
    preds_npy = preds['obs'].T.cpu().detach().numpy()
    np.save(f'{mcmc_mlp_path}/mcmc_pred.npy', preds_npy)
    return preds_npy  
    
    
def get_args_parse():
    parser = argparse.ArgumentParser(description="Liquid Water Content Prediction")
    parser.add_argument('--model-type', type=str, default="mlp", help='Model type')
    parser.add_argument('--method', type=str, default="mcmc", help="Posterior method: MCMC or SVI")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate for SVI")
    parser.add_argument('--mcmc_samples', type=int, default=260, help="num samples of MCMC")
    parser.add_argument('--num_epochs', type=int, default=1000, help="num epochs of SVI")
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args_parse()
    print(f"Model: {args.model_type} with {args.method}")
    use_cuda = True if torch.cuda.is_available() else False
    print(f"Use CUDA: {use_cuda}")
    
    if use_cuda:
        torch.set_default_device('cuda:1')
        args.device = 'cuda:1'
    else:
        args.device = 'cpu'
        
    X_train, X_test, y_train, y_test = create_dataset(to_tensor=True)
    X_train, X_test, y_train = X_train.to(args.device), X_test.to(args.device), y_train.to(args.device), 
    
    model = BNN(in_dim=7, out_dim=1, hid_dim=96, n_hid_layers=2, prior_scale=5.)
    model.to(args.device)
    
    if args.method == "svi":
        preds_npy = svi(model, X_train, y_train, X_test, args=args)
        # preds_npy = np.load(f'{svi_mlp_path}/svi_pred.npy')
        save_file=f"{svi_mlp_path}/svi-mlp.log"
        img_path = svi_mlp_path
    else:
        preds_npy = mcmc(model, X_train, y_train, X_test, args=args)
        # preds_npy = np.load(f'{mcmc_mlp_path}/mcmc_pred.npy')
        
        save_file=f"{mcmc_mlp_path}/mcmc-mlp.log"
        img_path = mcmc_mlp_path
        
    y_test, y_pred = de_normalize(y_test,  preds_npy)
    y_mean = y_pred.mean(axis=1)
    y_mean[y_mean < 0] = 0
    y_std = y_pred.std(axis=1)
    
    eval_all(y_test, y_mean, save_file=save_file)
    
    visualize_prediction(
        y_test, y_mean, y_pred_std=y_std, 
        img_path=img_path, interval=1)
    
    save_pred_csv(npy_data=y_mean, save_path=f"{mcmc_mlp_path}/mcmc_pred.csv")