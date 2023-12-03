import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_loader import RadarDataset, get_heights, get_input_dim
from data_configs import test_date_selection
from utils_read import data_mkdir


def plot_predictions(preds, index, img_path, type="mean", title='time'):
    plt.xlim([500, 10000])
    plt.ylim([-0.01, 0.08])
    plt.rcParams['figure.figsize'] = (14.4, 7.2)
    plt.xlabel("Height", fontsize=15)
    plt.ylabel("Liquid Water Content", fontsize=15)

    x = preds['height']
    y_label = preds['test_lwc'][index]

    if type == "mean":
        y_pred = preds['pred_lwc_mean'][index]
        y_std = preds['pred_lwc_std'][index]
        plt.fill_between(x, y_pred - y_std, y_pred + y_std, 
                         alpha=0.5, color='#9ac9db', zorder=5)
    else:
        y_pred = preds["pred_lwc"][index]

    plt.plot(x, y_label, 'o-', markersize=3, linewidth=3, 
             color="#F0988C", label="Micro-wave Radiometer")
    plt.plot(x, y_pred, '*-', markersize=6, linewidth=3, 
             color="#2878b5", label="Prediction")

    plt.legend(fontsize=10, frameon=False, loc='upper right')
    plt.title(title, fontsize=15)
    plt.grid()
    plt.savefig(f'{img_path}/{index}.png')
    plt.cla()


def get_test_data(test_date):
    # dates for test
    test_dataset = RadarDataset(dates=test_date)
    _, test_label = test_dataset.get_data_list()
    dim = test_dataset.num_heights
    length = test_dataset.num_timestamps
    input_data, _ = test_dataset[0]
    height = input_data[:, 0] * 10000

    return test_label, height, dim, length


def visualize_prediction(
        y_test,
        y_pred, 
        img_path,  
        y_pred_std=None,
        reshape=True,
        interval=1):
    
    from data_configs import data_path
    data_mkdir(img_path)
    heights = get_heights()
    num_timestamps, num_heights = get_input_dim()
    assert len(heights) == num_heights
        
    if y_pred_std is not None:    
        if reshape: # reshape back to [num_timestamps, num_heights]
            y_pred = y_pred.reshape((num_timestamps, num_heights))
            y_test = y_test.reshape((num_timestamps, num_heights))
            
        preds = {'height': heights,
                'test_lwc': y_test,
                'pred_lwc_mean': y_pred,
                'pred_lwc_std': y_pred_std}
        type_ = "mean"
        
    else:
        if reshape: # reshape back to [num_timestamps, num_heights]
            y_pred = y_pred.reshape((num_timestamps, num_heights))
            y_test = y_test.reshape((num_timestamps, num_heights))
        preds = {
            'height': heights,
            'test_lwc': y_test,
            'pred_lwc': y_pred}
        type_ = "standard"
        
    timestamps = []
    for _, date_ in enumerate(test_date_selection):
        test_df = pd.read_csv(f"{data_path}/lwc/{date_}_lwc.csv")
        timestamps.append(test_df.columns[1:])
    
    test_times = [item for sublist in timestamps for item in sublist]

    for test_idx in range(0, num_timestamps, interval):
        plot_predictions(
            preds=preds, img_path=img_path, index=test_idx, 
            type=type_, title=test_times[test_idx])