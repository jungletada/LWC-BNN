from sklearn.metrics import mean_squared_error, r2_score,\
    mean_absolute_error, explained_variance_score
import numpy as np


def clip(y):
    y[y < 0] = 0
    return y

    
def eval_all(y_true, y_pred, save_file=None):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    evs = explained_variance_score(y_true, y_pred)
    res = [
        "Mean Squared Error: %.5f" % mse,
        "Mean Absolute Error: %.5f" % mae,
        "R-squared Score: %.5f" % r2,
        "Explained Variance score Score: %.5f" % evs,
    ]
    for i in res:
        print(i)

    if save_file is not None:
        with open(save_file, 'w') as f:
            for i in res:
                f.write(i+'\n')
                
    return mse, mae, r2, evs