import os
import torch
import numpy as np
import pandas as pd
from typing import List
from datetime import datetime
from dateutil import parser
from data_configs import stat_save_path, data_path


def data_mkdir(directory_path):
    # 如果目录不存在，则创建目录
    if not os.path.exists(directory_path):
        os.makedirs(directory_path, exist_ok=True)


def walk_folder(folder_path):
    file_path = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path.append(os.path.join(root, file))
    return file_path


def output_time_stamp(
        csv_paths,
        save_path,
        format_='%Y/%m/%d %H:%M:%S'):
    """
    Select the time stamp in  the csv file
    :param csv_path: path to the csv file
    :param save_path: path to save the time stamp
    :param format_: datetime format
    :return: List[datetime]
    """
    times_stamps = []
    file = open(save_path, 'a')
    for csv_path in csv_paths:
        print(f"Dealing with {csv_path}")
        df = pd.read_csv(csv_path)
        time_strings = df.columns.tolist()[1:]
        for time_string in time_strings:
            file.write(time_string + '\n')
            times_stamps.append(parser.parse(time_string, fuzzy=True))
    file.close()
    return times_stamps


def find_closest_timestamp(
        target: datetime,
        timestamps: List[datetime]) -> datetime:
    """
    Find the timestamp in the list that is closest to the target timestamp.
    :param target: A datetime object representing the target timestamp
    :param timestamps: A list of datetime objects representing the timestamps to search through
    :return: The datetime object from the list that is closest to the target timestamp
    """
    closest_timestamp = None
    smallest_diff = None

    for timestamp in timestamps:
        time_diff = abs(target - timestamp)
        if smallest_diff is None or time_diff < smallest_diff:
            closest_timestamp = timestamp
            smallest_diff = time_diff
    return closest_timestamp


def de_normalize(y_true, y_pred):
    min_val = np.load(stat_save_path + 'lwc_min.npy')
    max_val = np.load(stat_save_path + 'lwc_max.npy')
    
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy()
        
    if torch.is_tensor(y_pred):
        y_pred = y_pred.cpu().numpy()
        
    y_true = y_true * (max_val - min_val) + min_val
    y_pred = y_pred * (max_val - min_val) + min_val
    
    return y_true, y_pred


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')