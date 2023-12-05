import os
import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
from utils_read import data_mkdir
from scipy.interpolate import interp1d
from sklearn.preprocessing import OneHotEncoder
from data_configs import stat_save_path, data_path
from data_configs import train_date_selection, test_date_selection


def min_max_scale(array, keyword, to_tensor=False):
    data_mkdir(stat_save_path)
    p_min = stat_save_path + f'{keyword}_min.npy'
    p_max = stat_save_path + f'{keyword}_max.npy'

    if os.path.exists(p_min) and os.path.exists(p_max):
        min_val = np.load(stat_save_path + f'{keyword}_min.npy')
        max_val = np.load(stat_save_path + f'{keyword}_max.npy')
    else:
        min_val, max_val = array.min(), array.max()
        np.save(p_min, min_val)
        np.save(p_max, max_val)

    normalized = (array - min_val) / (max_val - min_val)

    if to_tensor:
        return torch.tensor(normalized, dtype=torch.float32)
    else:
        return normalized


def encode_time(timestamps):
    # Extract the hour from each timestamp and store it in a list
    hours = [pd.to_datetime(ts).hour for ts in timestamps]
    # Reshape the hours list to a numpy array for one hot encoding
    hours_array = np.array(hours).reshape(-1, 1)
    # Initialize the OneHotEncoder
    encoder = OneHotEncoder(categories=[range(24)])
    # Fit and transform the data to one hot encoding
    one_hot_encoded_hours = encoder.fit_transform(hours_array).toarray()
    return one_hot_encoded_hours


def filter_height(df, min_h=600, max_h=10000):
    filtered_df = df[(df['Height'] >= min_h) & (df['Height'] <= max_h)]
    return filtered_df


def interp_function(original_indices, original_vector):
    return interp1d(original_indices, original_vector, kind='linear')
    
    
class RadarDataset(Dataset):
    def __init__(self, dates, root_dir=data_path, filter_h=False, to_tensor=False):
        """
        Custom dataset to load data from multiple CSV files.
        :param dates: A list of date strings to load the data for.
        :param root_dir: The root directory where the CSV files are stored.
        """
        self.filter_h = filter_h
        self.num_inputs = 7 
        self.k, self.w, self.p, self.t, self.h = [], [], [], [], []
        self.lwc, self.time = [], []
        self.to_tensor = to_tensor

        for date in dates: # Input data file paths
            # ka-band 和 w-band
            ka_band_path = root_dir + f"ka_band/{date}_ka_band.csv"
            w_band_path = root_dir + f"w_band/{date}_w_band.csv"
            # 温度，气压，湿度
            pressure_path = root_dir + f"pressure/{date}_pressure.csv"
            relative_humidity_path = root_dir + f"relativeHumidity/{date}_relativeHumidity.csv"
            temperature_path = root_dir + f"temperature/{date}_temperature.csv"
            # Label data file path
            lwc_path = root_dir + f"lwc/{date}_lwc.csv"

            try:
                # Load input data if available, otherwise use an empty DataFrame
                ka_band_df = pd.read_csv(ka_band_path) if os.path.exists(ka_band_path) else pd.DataFrame()
                w_band_df = pd.read_csv(w_band_path) if os.path.exists(w_band_path) else pd.DataFrame()
                pressure_df = pd.read_csv(pressure_path) if os.path.exists(pressure_path) else pd.DataFrame()
                humidity_df = pd.read_csv(relative_humidity_path) if os.path.exists(
                    relative_humidity_path) else pd.DataFrame()
                temperature_df = pd.read_csv(temperature_path) if os.path.exists(temperature_path) else pd.DataFrame()
                # Load labels
                lwc_df = pd.read_csv(lwc_path) if os.path.exists(lwc_path) else pd.DataFrame()
                
                if self.filter_h:
                    ka_band_df = filter_height(ka_band_df)
                    w_band_df = filter_height(w_band_df)
                    pressure_df = filter_height(pressure_df)
                    humidity_df = filter_height(humidity_df)
                    temperature_df = filter_height(temperature_df)
                    lwc_df = filter_height(lwc_df)
                
                self.heights = lwc_df['Height'].values / 10000.0
                self.time.append(lwc_df.columns[1:])
                
                self.k.append(ka_band_df.iloc[:, 1:].values)
                self.w.append(w_band_df.iloc[:, 1:].values)
                self.p.append(pressure_df.iloc[:, 1:].values)
                self.t.append(temperature_df.iloc[:, 1:].values)
                self.h.append(humidity_df.iloc[:, 1:].values)
                self.lwc.append(lwc_df.iloc[:, 1:].values)
                # print(f"- Data for {date}: {len(lwc_df.columns) - 1} timestamps.")

            except Exception as e:
                print(f"Could not load data for date {date}: {str(e)}")
            print(f"| {date} | {len(lwc_df.columns) - 1} |")

        # Concatenate all the data and labels into single tensors
        self.k = np.concatenate(self.k, axis=1).transpose(1, 0)
        self.w = np.concatenate(self.w, axis=1).transpose(1, 0)
        
        self.k = min_max_scale(self.k, 'k')
        self.w = min_max_scale(self.w, 'w')
        
        self.diff = self.k - self.w
        self.diff = min_max_scale(self.diff, 'diff')
        
        self.p = np.concatenate(self.p, axis=1).transpose(1, 0)
        self.p = min_max_scale(self.p, 'p')

        self.t = np.concatenate(self.t, axis=1).transpose(1, 0)
        self.t = min_max_scale(self.t, 't')

        self.h = np.concatenate(self.h, axis=1).transpose(1, 0)
        self.h = min_max_scale(self.h, 'h')

        self.lwc = np.concatenate(self.lwc, axis=1).transpose(1, 0)
        self.lwc = min_max_scale(self.lwc, 'lwc')

        self.num_timestamps, self.num_heights = self.lwc.shape
        self.total_len = self.num_heights * self.num_timestamps
        
        self.time = np.concatenate(self.time, axis=0)
        self.time = encode_time(self.time)
        height_range = np.linspace(0, 1, self.num_heights)
        time_range = np.linspace(0, 1, 24)
        interpolate_func = interp1d(time_range, self.time, kind='linear')
        self.inter_time = interpolate_func(height_range)
        
        self.input_list, self.label_list = self.get_data_list()
        self.input_list = self.input_list.transpose(0, 2, 1).reshape((-1, self.num_inputs))
        self.label_list = self.label_list.reshape(-1, 1)
        
    def __len__(self):
        """
        Return the total number of samples in the dataset.
        """
        return self.total_len

    def __getitem__(self, idx):
        input_data = self.input_list[idx]
        label = self.label_list[idx]
        
        if self.to_tensor:
            return  torch.tensor(input_data, dtype=torch.float32), \
                    torch.tensor(label, dtype=torch.float32), 
        return input_data, label
        
    def get_data_list(self):
        height = np.expand_dims(self.heights, axis=0)
        height = np.tile(height, (self.num_timestamps, 1))
        input_list = np.stack((
            height, self.k, self.w, self.diff, self.p, self.t, self.h), axis=1)
        label_list = self.lwc
        return input_list, label_list


def create_dataset(to_tensor=False):
    from data_loader import RadarDataset
    from data_configs import train_date_selection, test_date_selection
    
    train_dataset = RadarDataset(# Define train data
        dates=train_date_selection,
        filter_h=False,
        to_tensor=to_tensor)
    X_train, y_train = train_dataset.get_data_list()
    
    test_dataset = RadarDataset(# Define test data
        dates=test_date_selection,
        filter_h=True,
        to_tensor=to_tensor)
    X_val, y_val = test_dataset.get_data_list()
    
    num_train, dim, num_heights = X_train.shape
    print(f"{num_train} data for training, with {num_heights} heights.")
    
    X_train = X_train.transpose(0, 2, 1).reshape((-1, dim))
    X_val = X_val.transpose(0, 2, 1).reshape((-1, dim))
    
    y_train = y_train.reshape((-1))
    y_val = y_val.reshape((-1))
    
    num_test, num_heights = X_val.shape
    print(f"{num_test} data for testing")
    print(f"X_train: {X_train.shape}, y_train:{y_train.shape}\nX_val:{X_val.shape}, y_val:{y_val.shape}")
    
    if to_tensor:
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        X_val = torch.tensor(X_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.float32)
    
    return X_train, X_val, y_train, y_val


def get_heights(scale_back=True):
    testset = RadarDataset(test_date_selection,filter_h=True)
    if scale_back:
        return testset.heights * 10000
    return testset.heights


def get_input_dim():
    testset = RadarDataset(test_date_selection,filter_h=True)
    return testset.num_timestamps, testset.num_heights
      
    
if __name__ == '__main__':
    trainset = RadarDataset(
        train_date_selection,
        root_dir=data_path, to_tensor=True)
    testset = RadarDataset(
        test_date_selection,
        root_dir=data_path)
    input_, label_ = trainset[10]
    print(f"Shape of Input: {input_.shape}")
    print(f"Shape of Label: {label_.shape}")
    print(f"#Train data: {len(trainset)}")
    print(f"#Test data: {len(testset)}")
    
    input_list, label_list = trainset.input_list, trainset.label_list
    print(f"Shape of input_list: {input_list.shape}")
    print(f"Shape of label_list: {label_list.shape}")