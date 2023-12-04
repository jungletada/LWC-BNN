# import some basic libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor

from data_loader import create_dataset
from utils_read import de_normalize
from evaluate import clip, eval_all
from visualize import visualize_prediction
import warnings
# Ignore all warnings
warnings.filterwarnings("ignore")


def main_run(X_train,X_val,y_train,y_val):
    # Find Which Model Perform better 
    models = [
            ("random forest", RandomForestRegressor()),
            ("linear regression", LinearRegression()),
            ("xgboost", XGBRegressor()),
            ("decision tree", DecisionTreeRegressor())]

    # Metrics
    names = []
    maes, mses, r2_scores, ev_scores= [], [], [], []

    for (name, model) in models:
        print(f"------------- processing model {name}. -------------")
        pipe = Pipeline([
            (("model", model))])
        
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_val)
            
        y_val, y_pred = de_normalize(y_val, y_pred)
        y_val, y_pred = clip(y_val), clip(y_pred)
        
        visualize_prediction(y_val, y_pred, img_path=f"results/{name}/", interval=1)
        mse, mae, r2, evs = eval_all(y_val, y_pred, save_file=f"results/{name}/{name}.log")
        print(f"------------- processing finished. -------------")
        mses.append(mse)
        maes.append(mae)
        r2_scores.append(r2)
        ev_scores.append(evs)
        names.append(name)
        
    return mses, maes, r2_scores, ev_scores, names


def plot(mses, maes, r2_scores, ev_scores, names):
    plt.figure(figsize=(10, 4))
    plt.plot(names, mses, marker='s',linestyle=':',color='k',label="MSE")
    plt.plot(names, r2_scores, marker='o',linestyle='-',color='b',label="R2")
    plt.plot(names, maes, marker='s', linestyle=':',color='g',label="MAE")
    plt.plot(names, ev_scores, marker='s',linestyle=':',color='y',label="EV")
    
    for i in range(len(names)):
        plt.text(names[i], mses[i], "{:.2e}".format(mses[i]), ha='right', va='top')
        plt.text(names[i], r2_scores[i], "{:.2f}".format(r2_scores[i]), ha='right', va='bottom')
        plt.text(names[i], maes[i], "{:.2e}'".format(maes[i]), ha='right', va='top')
        plt.text(names[i], ev_scores[i], "{:.2f}".format(ev_scores[i]), ha='right', va='bottom')
        
    plt.legend()
    plt.savefig("results/ML-methods.png")
    

if __name__ == '__main__':
    X_train,X_val,y_train,y_val = create_dataset()
    mses, maes, r2_scores, ev_scores, names = main_run(X_train,X_val,y_train,y_val)
    plot(mses, maes, r2_scores, ev_scores, names)