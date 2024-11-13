import os
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from dataset import MLdataset
from models.model import InformerStack
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error

import warnings
warnings.filterwarnings('ignore')


from accelerate import Accelerator
from datetime import datetime
from colorama import Fore, Back, Style
r_ = Fore.RED
b_ = Fore.BLUE
c_ = Fore.CYAN
g_ = Fore.GREEN
y_ = Fore.YELLOW
m_ = Fore.MAGENTA
sr_ = Style.RESET_ALL


def plot_and_save_result(time, real, predicted, filename):
    plt.figure(figsize=(10, 8))
    plt.plot(time, real, label='Ground Truth', linestyle='-')
    plt.plot(time, predicted, label='predicted value', linestyle='--')
    plt.gcf().autofmt_xdate()
    plt.xlabel('Date')
    plt.ylabel('Temperature')
    plt.title("Comparision real vs predicted")
    plt.tight_layout()
    plt.legend()
    plt.savefig(filename)
    plt.close()


def inference(load_path, save_path, model, dl, criterion):
    inference_loss = 0
    true_labels = []
    pred_labels = []
    time_list = []
    total = 0.0
    label_len = 0
    model.eval()
    with torch.no_grad():
        for i, (data, labels) in enumerate(dl):
            inputs_t = data[:,:,:4]
            batch_s, window, date = inputs_t.shape
            inputs = data[:,:,4:]
            
            total += inputs.shape[0] * inputs.shape[1]

            output = model(inputs,inputs_t,inputs,inputs_t)
            
            labels = labels.to(output.device)
            label_len = labels.shape[1]
                
            loss = criterion(output, labels)
            
            inference_loss += loss.item()
            
            inputs_t = inputs_t.cpu().numpy().reshape(-1,date)
            time_list.append([datetime(int(item[0]),int(item[1]),int(item[2]),int(item[3]))for item in inputs_t])
            true_labels.append(labels.cpu().numpy())
            pred_labels.append(output.cpu().numpy())

        inference_loss /= total
        
        
        metrics = {}
        time_list = np.concatenate(time_list, axis=0).reshape(-1,label_len)
        true = np.concatenate(true_labels, axis=0).reshape(-1,label_len)
        pred = np.concatenate(pred_labels, axis=0).reshape(-1,label_len)
        
        # 성능 지표 계산
        name = 'Temperature'
        mse = mean_squared_error(true, pred)
        mae = mean_absolute_error(true, pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(true, pred)
        medae = median_absolute_error(true, pred)
        num = np.random.randint(0, len(dl), size=1)
        model_name = load_path.split('\\')[-1][:-3]
        plot_and_save_result(time_list[num][0], true[num][0], pred[num][0], filename=os.path.join(save_path,f'{model_name}_inference.png'))
        
        metrics[name] = {'MSE': mse, 'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MedAE': medae}


        # 성능 지표 출력
        print(f"\nMetrics for {name}_output:")
        print(f"MSE: {metrics[name]['MSE']:.4f}")
        print(f"MAE: {metrics[name]['MAE']:.4f}")
        print(f"RMSE: {metrics[name]['RMSE']:.4f}")
        print(f"R2: {metrics[name]['R2']:.4f}")
        print(f"MedAE: {metrics[name]['MedAE']:.4f}")
        print('-' * 50)
        
        return inference_loss, metrics



def make_result(load_exp, save_path, model, dl, criterion):
    inference_loss = 0
    true_labels = []
    pred_labels = []
    time_list = []
    total = 0.0
    label_len = 0
    model.eval()
    with torch.no_grad():
        for i, (data, labels) in enumerate(dl):
            inputs_t = data[:,:,:4]
            batch_s, window, date = inputs_t.shape
            inputs = data[:,:,4:]
            
            total += inputs.shape[0] * inputs.shape[1]

            output = model(inputs,inputs_t,inputs,inputs_t)
            
            labels = labels.to(output.device)
            label_len = labels.shape[1]
                
            loss = criterion(output, labels)
            
            inference_loss += loss.item()
            
            inputs_t = inputs_t.cpu().numpy().reshape(-1,date)
            time_list.append([datetime(int(item[0]),int(item[1]),int(item[2]),int(item[3]))for item in inputs_t])
            true_labels.append(labels.cpu().numpy())
            pred_labels.append(output.cpu().numpy())

        inference_loss /= total
        
        
        metrics = {}
        time_list = np.concatenate(time_list, axis=0).reshape(-1,label_len)
        true = np.concatenate(true_labels, axis=0).reshape(-1,label_len)
        pred = np.concatenate(pred_labels, axis=0).reshape(-1,label_len)
        
        # 성능 지표 계산
        name = 'Temperature'
        mse = mean_squared_error(true, pred)
        mae = mean_absolute_error(true, pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(true, pred)
        medae = median_absolute_error(true, pred)
        num = np.random.randint(0, len(dl), size=1)
        plot_and_save_result(time_list[num][0], true[num][0], pred[num][0], filename=os.path.join(save_path,"inference.png"))
        
        metrics[name] = {'MSE': mse, 'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MedAE': medae}


        # 성능 지표 출력
        print(f"\nMetrics for {name}_output:")
        print(f"MSE: {metrics[name]['MSE']:.4f}")
        print(f"MAE: {metrics[name]['MAE']:.4f}")
        print(f"RMSE: {metrics[name]['RMSE']:.4f}")
        print(f"R2: {metrics[name]['R2']:.4f}")
        print(f"MedAE: {metrics[name]['MedAE']:.4f}")
        print('-' * 50)
        
        return inference_loss, metrics


if __name__ == '__main__':
    # torch.set_default_dtype(torch.float32)
    # run setting Date setting
    now = datetime.now()
    exp_date = now.strftime('%m%d-%S%M')
    
    # device Setting
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    # config
    # data config
    
    # inferencing config
    load_exp = '1106-0545'
    project_name = 'ML-TeamProject'
    batch_size = 128
    mode = 'train'
    window = 24*7
    hop = 0
    enc_in = dec_in = 16    # 인코더, 디코더 입력 차원 수 (예: 피처 수 16)
    c_out = 1               # 출력 차원 수 (예: 예측할 온도 값의 차원 1)
    seq_len = label_len = out_len = window # 인코더, 디코더 입력 시퀀스 길이, 출력 시퀀스 길이
    
    ############## loading data ###################
    path = r'F:\MLProject\data\train.csv'
    dataset = MLdataset(path, mode, window, hop)
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    load_path = os.path.join(os.getcwd(),'saved_Model',f'{project_name}_result',load_exp,f'{load_exp}_BestTrainLoss_informer.pt')
    model = InformerStack(enc_in - 4,
                            dec_in - 4,
                            c_out,
                            seq_len,
                            label_len,
                            out_len,
                            dropout=0.2
                            )
    model.load_state_dict(torch.load(load_path))
    criterion = nn.MSELoss()
    
    accelerator = Accelerator()
    print(f"{accelerator.device} is used")
    model,dl,criterion = accelerator.prepare(model,dl,criterion)
    

    # Inference_Result dir Create
    save_path = os.path.join(os.getcwd(),'Inference_Result')
    if os.path.isdir(save_path):
        print('Path alreay exist')
    else:
        os.mkdir(save_path)
    # experiment dir create
    save_path = os.path.join(save_path, load_exp)
    if os.path.isdir(save_path):
        print('Path alreay exist')
    else:
        os.mkdir(save_path)
    
    best_loss = 9999999
    start_time = time.time()
    print(f"{project_name}\nInference Started")
    inference_loss, metrics = inference(load_path, save_path, model, dl, criterion)
    end_time = time.time()
    print(f'{end_time - start_time} time executed..\n')
    
    
    ############## loading data ###################
    path = r'F:\MLProject\data\train.csv'
    dataset = MLdataset(path, mode, window, hop)
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    load_path = os.path.join(os.getcwd(),'saved_Model',f'{project_name}_result',load_exp,f'{load_exp}_BestValidLoss_informer.pt')
    model = InformerStack(enc_in - 4,
                            dec_in - 4,
                            c_out,
                            seq_len,
                            label_len,
                            out_len,
                            dropout=0.2
                            )
    model.load_state_dict(torch.load(load_path))
    criterion = nn.MSELoss()
    
    accelerator = Accelerator()
    print(f"{accelerator.device} is used")
    model,dl,criterion = accelerator.prepare(model,dl,criterion)
    

    # Inference_Result dir Create
    save_path = os.path.join(os.getcwd(),'Inference_Result')
    if os.path.isdir(save_path):
        print('Path alreay exist')
    else:
        os.mkdir(save_path)
    # experiment dir create
    save_path = os.path.join(save_path, load_exp)
    if os.path.isdir(save_path):
        print('Path alreay exist')
    else:
        os.mkdir(save_path)
    
    best_loss = 9999999
    start_time = time.time()
    print(f"{project_name}\nInference Started")
    inference_loss, metrics = inference(load_path, save_path, model, dl, criterion)
    end_time = time.time()
    print(f'{end_time - start_time} time executed..\n')
    
    