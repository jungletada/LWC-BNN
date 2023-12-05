import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_loader import RadarDataset, get_heights, get_input_dim
from data_configs import test_date_selection
from utils_read import data_mkdir
from data_configs import data_path


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
        

def save_pred_csv(npy_data, save_path):
    idx = 0
    num_h = 41
    for _, date_ in enumerate(test_date_selection):
        csv_data  = pd.read_csv(f"{data_path}/lwc/{date_}_lwc.csv")
        num_times = len(csv_data.columns[1:])
        d_data = npy_data[idx: (idx+num_times) * num_h]
        d_data = d_data.reshape(num_h, -1)
        npy_data_df = pd.DataFrame(d_data, columns=csv_data.columns[1:])
        npy_data_df.insert(0, 'Height', csv_data['Height'])
        npy_data_df.to_csv(save_path, index=False)
        idx = (idx+num_times) * num_h


def plot_ground_truth_vs_prediction(height, ground_truth_data, prediction_data):
    """
    Plots the ground truth vs predictions for a specified height.

    :param height: The height for which the data is to be plotted.
    :param ground_truth_data: DataFrame containing the ground truth data.
    :param prediction_data: DataFrame containing the prediction data.
    """
    # Filter data for the specified height
    ground_truth_height = ground_truth_data[ground_truth_data['Height'] == height].iloc[0, 1:]
    prediction_height = prediction_data[prediction_data['Height'] == height].iloc[0, 1:]

    # Extracting timestamps from columns (assuming both dataframes have the same timestamp columns)
    timestamps = ground_truth_data.columns[1:]

    # Plotting
    plt.figure(figsize=(15, 7))
    plt.plot(timestamps, ground_truth_height, label='Ground Truth', marker='o')
    plt.plot(timestamps, prediction_height, label='Prediction', marker='x')

    # Formatting the plot
    plt.xlabel('Timestamp')
    plt.ylabel('Values')
    plt.title(f'Ground Truth vs Prediction for Height {height}')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Show the plot
    plt.savefig(f"figures/random-forest-{height}.png")
    plt.close()



def save_to_excel():
    
    def get_results(string):
        import re
        # 使用正则表达式匹配浮点数
        match = re.search(r'\d+\.\d+', string)
        if match:
            # 将匹配的字符串转换为浮点数
            number = float(match.group())
            return number
        else:
            print("No floating point number found.")

    # Data to be included in the Excel file
    names = ["mcmc-mlp", "random-forest", "linear-regression", "decision-tree", "xgboost"]
    mse, mae, rs, evs = [], [], [], []
    for name in names:
        log_path = f"results/{name}/{name}.log"
        with open (log_path, "r") as f:
            lines = f.readlines()
            mse.append(get_results(lines[0]))
            mae.append(get_results(lines[1]))
            rs.append(get_results(lines[2]))
            evs.append(get_results(lines[3]))
                
    data = {
        "Model": names,
        "Mean Squared Error": mse,
        "Mean Absolute Error": mae,
        "R-squared Score": rs,
        "Explained Variance Score": evs,
    }

    # Creating a DataFrame from the data
    df = pd.DataFrame(data)
    # Setting the 'Model' column as the index
    df.set_index("Model", inplace=True)
    print(df)
    # Save the DataFrame to an Excel file
    excel_file_path = 'results/model_evaluation.xlsx'
    df.to_excel(excel_file_path)
    excel_file_path


if __name__ == '__main__':
    # for _, date_ in enumerate(test_date_selection):
    #     csv_data  = pd.read_csv(f"{data_path}/lwc/{date_}_lwc.csv")
    #     pred_data = pd.read_csv("results/random-forest/random-forest_pred.csv")
    #     for height in csv_data["Height"]:
    #         plot_ground_truth_vs_prediction(height, csv_data, pred_data)
    save_to_excel()
