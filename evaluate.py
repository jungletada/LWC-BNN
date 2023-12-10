from sklearn.metrics import mean_squared_error, r2_score,\
    mean_absolute_error, explained_variance_score
import datetime   
import numpy as np
import pandas as pd


def clip(y):
    y[y < 0] = 0
    return y

    
def eval_all(y_true, y_pred, save_file=None):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    evs = explained_variance_score(y_true, y_pred)
    res = [
        # "Mean Squared Error: %.5f" % mse,
        "Mean Absolute Error: %.5f" % mae,
        "R-squared Score: %.5f" % r2,
        "Explained Variance Score: %.5f" % evs,
    ]
    for i in res:
        print(i)

    if save_file is not None:
        with open(save_file, 'w') as f:
            f.write(save_file+'\n')
            for i in res:
                f.write(i+'\n')
                
    return mse, mae, r2, evs


def clip_time(df, save_to):
    # Convert the column names to datetime objects for filtering
    # Excluding the first column 'Height' which is not a timestamp
    timestamps = pd.to_datetime(df.columns[1:], format='%Y/%m/%d %H:%M:%S')

    # Define the start and end time for filtering
    start_time = datetime.time(17, 12)
    end_time = datetime.time(20, 20)

    # Filter the timestamps that fall between 17:20 and 20:20
    filtered_timestamps = [time for time in timestamps if start_time <= time.time() <= end_time]

    # Filter the DataFrame to include only the columns with the selected timestamps
    # Adding the 'Height' column to the list of filtered columns
    if save_to is not None:
        save_columns = ["Height"] + [time.strftime('%Y/%m/%d %H:%M:%S') for time in filtered_timestamps]
        save_df = df[save_columns]
        save_df.to_csv(save_to, index=False)
    
    filtered_columns = [time.strftime('%Y/%m/%d %H:%M:%S') for time in filtered_timestamps]
    filtered_data = df[filtered_columns]

    return filtered_data

if __name__ == '__main__':
    models = ['decision-tree', 'linear-regression', 'mcmc-mlp', 'xgboost']
    # models = ['mcmc-mlp']
    for model in models:
        print(f"{model}:")
        pred_df = pd.read_csv(f"results/{model}/{model}_pred.csv")
        # pred_df = pd.read_csv("results/mcmc-mlp/mcmc-mlp_std.csv")
        label_df = pd.read_csv("data-slim/lwc/20130515_lwc.csv")
        
        pred_filter = clip_time(pred_df, save_to=f"paper_results/{model}.csv").to_numpy()
        pred_filter = pred_filter.reshape(-1)
        
        label_filter = clip_time(label_df, save_to=None).to_numpy()
        label_filter = label_filter.reshape(-1)
        
        eval_all(label_filter, pred_filter, save_file=f'paper_results/{model}.log')