import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def get_data(path, mode):
    csv_data = pd.read_csv(path)
    columns = csv_data.columns.values
    columns = columns[:-5]
    columns = np.delete(columns,np.where(columns == 'Wind Direction (deg)'))
    columns = np.delete(columns,np.where(columns == 'Cloud Height 3rd Layer (FT)'))
    
    # 결측치 제거
    if mode == 'train':
        csv_data = csv_data[columns]
        print("{0} == 0 data num: {1}".format(columns[7],len(csv_data[csv_data[columns[7]] == 0].index)))
        csv_data = csv_data.replace({columns[7]:0},csv_data.mean()[columns[7]])
        print("{0} == 0 data num: {1}".format(columns[7],len(csv_data[csv_data[columns[7]] == 0].index)))
        print()
        
        print("{0} > 8 data num: {1}".format(columns[9],len(csv_data[csv_data[columns[9]] > 8].index)))
        csv_data = csv_data.replace({columns[9]:9},8)
        print("{0} > 8 data num: {1}".format(columns[9],len(csv_data[csv_data[columns[9]] > 8].index)))
        print()

        print("{0} == 0 data num: {1}".format(columns[11:13],len(csv_data[csv_data[columns[11]] == 0].index) + len(csv_data[csv_data[columns[12]] == 0].index)))
        csv_data = csv_data.replace({columns[11]:0,columns[12]:0},csv_data.mean()[columns[11:13]])
        print("{0} == 0 data num: {1}".format(columns[11:13],len(csv_data[csv_data[columns[11]] == 0].index) + len(csv_data[csv_data[columns[12]] == 0].index)))
        print()

        print("{0} > 30 data num: {1}".format(columns[13],len(csv_data[csv_data[columns[13]] > 30].index)))
        normal_mean = np.mean(csv_data[csv_data[columns[13]] <= 30][columns[13]].unique())
        csv_data.loc[csv_data[columns[13]] > 30, columns[13]] = normal_mean
        print("{0} > 30 data num: {1}".format(columns[13],len(csv_data[csv_data[columns[13]] > 30].index)))
        print()
    elif mode == 'valid' or mode == 'validation':
        csv_data = csv_data[columns]
        print("{0} == 0 data num: {1}".format(columns[6],len(csv_data[csv_data[columns[6]] == 0].index)))
        csv_data = csv_data.replace({columns[6]:0},csv_data.mean()[columns[6]])
        print("{0} == 0 data num: {1}".format(columns[6],len(csv_data[csv_data[columns[6]] == 0].index)))
        print()
        
        print("{0} > 8 data num: {1}".format(columns[8],len(csv_data[csv_data[columns[8]] > 8].index)))
        csv_data = csv_data.replace({columns[8]:9},8)
        print("{0} > 8 data num: {1}".format(columns[8],len(csv_data[csv_data[columns[8]] > 8].index)))
        print()

        print("{0} == 0 data num: {1}".format(columns[10:12],len(csv_data[csv_data[columns[10]] == 0].index) + len(csv_data[csv_data[columns[11]] == 0].index)))
        csv_data = csv_data.replace({columns[10]:0,columns[11]:0},csv_data.mean()[columns[10:12]])
        print("{0} == 0 data num: {1}".format(columns[10:12],len(csv_data[csv_data[columns[10]] == 0].index) + len(csv_data[csv_data[columns[11]] == 0].index)))
        print()

        print("{0} > 30 data num: {1}".format(columns[12],len(csv_data[csv_data[columns[12]] > 30].index)))
        normal_mean = np.mean(csv_data[csv_data[columns[12]] <= 30][columns[12]].unique())
        csv_data.loc[csv_data[columns[12]] > 30, columns[12]] = normal_mean
        print("{0} > 30 data num: {1}".format(columns[12],len(csv_data[csv_data[columns[12]] > 30].index)))
        print()
    
    return csv_data, columns

def minmaxScale(data,columns):
    scaler = MinMaxScaler()
    data[columns] = scaler.fit_transform(data[columns])
    return data

def stdScale(data,columns):
    scaler = StandardScaler()
    data[columns] = scaler.fit_transform(data[columns])
    return data

def interpol(data, columns):
    data = data.drop(columns=['ID'])
    not_interpolate = ['Year','Month','Day','Hour','ID','Weather Phenomenon (null)','Cloud Type 1st Layer (null)','Total Cloud Cover (1/8)']
    interpol = [i for i in columns if i not in not_interpolate]
    del not_interpolate[:5]
    
    data['Datetime'] = pd.to_datetime(data[['Year', 'Month', 'Day', 'Hour']])
    data.set_index('Datetime', inplace=True)
    
    data = data.resample('h').asfreq()
    data[interpol] = data[interpol].interpolate(method='time')
    data[not_interpolate] = data[not_interpolate].ffill()
    data['Year'] = data.index.year
    data['Month'] = data.index.month
    data['Day'] = data.index.day
    data['Hour'] = data.index.hour
    
    return data

def windowed(data, mode, window, hop):
    if mode == 'train':
        X, y = [], []
        temper = data['Temperature']
        data = data.drop(columns=['Temperature'])
        for i in range(len(data) - window):
            X.append(data[i:i+window].to_numpy())
            y.append(temper[i:i+window].to_numpy())
        return np.array(X), np.array(y)
    elif mode == 'valid' or mode == 'validation':
        X = []
        for i in range(len(data) - window):
            X.append(data[i:i+window].to_numpy())
        return np.array(X)

def prepare(data, columns, mode, window, hop):
    if mode == 'train':
        minmax_column = [columns[9],columns[17]]
        std_column = [columns[6],columns[7],columns[8],
                    columns[10],columns[11],columns[12],
                    columns[13],columns[14],columns[15],
                    columns[16],columns[17]]
    elif mode == 'valid' or mode == 'validation':
        minmax_column = [columns[8],columns[16]]
        std_column = [columns[5],columns[6],columns[7],
                    columns[9],columns[10],columns[11],
                    columns[12],columns[13],columns[14],
                    columns[15],columns[16]]
        
    data = minmaxScale(data, minmax_column)
    # data = stdScale(data, std_column)
    data = interpol(data, columns)
    data = windowed(data, mode, window, hop)
    return data
    


class MLdataset(Dataset):
    def __init__(self,path, mode, window, hop):
        super(MLdataset).__init__()
        self.path = path
        self.mode = mode
        self.window = window
        self.hop = hop
        self.data, self.columns = get_data(self.path, self.mode)
        self.data = prepare(self.data, self.columns, self.mode, self.window, self.hop)
        
    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, index):
        if self.mode == 'train':
            d = torch.tensor(self.data[0][index], dtype=torch.float32)
            l = self.data[1][index].reshape(self.window,-1)
            l = torch.tensor(l,dtype=torch.float32)
            return d, l
        elif self.mode == 'valid' or self.mode == 'validation':
            return self.data[index]

if __name__ == '__main__':
    path = r'/workspace/MLProject/data/train.csv'
    mode = 'valid'
    window = 24*20
    hop = 6
    dataset = MLdataset(path, mode, window, hop)
    
    batch_size = 1
    
    debug_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for batch_idx, data in enumerate(debug_loader):
        if batch_idx % 1 == 0:
            # print(data[0])
            print(data.shape)
            print(data[:,:,:4].shape)
            print(data[:,:,4:].shape)
            # print(label.shape)
            # print(label)
            exit()