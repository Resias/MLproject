import os
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from dataset import MLdataset
from models.model import InformerStack, Informer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error

import warnings
warnings.filterwarnings('ignore')

import wandb

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


def evaluate(model,valid_loader,criterion):
    model.eval()
    valid_loss = 0
    true_labels = []
    pred_labels = []
    with torch.no_grad():
        for i, (data, labels) in enumerate(valid_loader):
            inputs_t = data[:,:,:4]
            inputs = data[:,:,4:]
            
            half_point_t = inputs_t.shape[2]//2
            half_point_d = inputs_t.shape[2]//2
            dec_t = torch.zeros_like(inputs_t).to(device=inputs_t.device)
            dec_d = torch.zeros_like(inputs).to(device=inputs.device)
            
            dec_t[:, :, :half_point_t] = inputs_t[:, :, :half_point_t]
            dec_d[:, :, :half_point_d] = inputs[:, :, :half_point_d]

            output = model(inputs,inputs_t,dec_d,dec_t)
            
            half_point_l = labels.shape[1]//2
            labels = labels.to(output.device)
                
            loss = criterion(output[:,:half_point_l], labels[:,half_point_l:])
            valid_loss += loss.item()
            
            
            true_labels.append(labels[:,half_point_l:].cpu().numpy().reshape(-1,1))
            pred_labels.append(output[:,:half_point_l].cpu().numpy().reshape(-1,1))
                
    
    metrics = {}
    true = np.concatenate(true_labels, axis=0)
    pred = np.concatenate(pred_labels, axis=0)
    
    # print(true.shape)
    # print(true[0])
    # print(pred.shape)
    # print(pred[0])
    # exit()
    
    # 성능 지표 계산
    name = 'Temperature'
    mse = mean_squared_error(true, pred)
    mae = mean_absolute_error(true, pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(true, pred)
    medae = median_absolute_error(true, pred)
    
    metrics[name] = {'MSE': mse, 'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MedAE': medae}

    # 성능 지표 출력
    print(f"\nMetrics for {name}_output:")
    print(f"MSE: {metrics[name]['MSE']:.4f}")
    print(f"MAE: {metrics[name]['MAE']:.4f}")
    print(f"RMSE: {metrics[name]['RMSE']:.4f}")
    print(f"R2: {metrics[name]['R2']:.4f}")
    print(f"MedAE: {metrics[name]['MedAE']:.4f}")
    print('-' * 50)
    
    valid_loss /= len(valid_loader)
    return valid_loss

def train_and_evaluate_loop(train_loader,valid_loader,model,optimizer,
                            criterion,epoch,best_t_loss, best_v_loss,
                            total_epoch,
                            save_path,exp_date,lr_scheduler=None):
    train_loss = 0
    model.train()
    for i, (data, label) in enumerate(train_loader):
        optimizer.zero_grad()
        
        inputs_t = data[:,:,:4]
        inputs = data[:,:,4:]
        half_point_t = inputs_t.shape[2]//2
        half_point_d = inputs_t.shape[2]//2
        dec_t = torch.zeros_like(inputs_t).to(device=inputs_t.device)
        dec_d = torch.zeros_like(inputs).to(device=inputs.device)
        
        dec_t[:, :, :half_point_t] = inputs_t[:, :, :half_point_t]
        dec_d[:, :, :half_point_d] = inputs[:, :, :half_point_d]

        output = model(inputs,inputs_t,dec_d,dec_t)
        
        half_point_l = label.shape[1]//2
        label = label.to(output.device)
        # print(label.shape)
        # print(output.shape)
        # exit()
            
        loss = criterion(output[:,:half_point_l], label[:,half_point_l:])

        accelerator.backward(loss) 
        optimizer.step()
        train_loss += loss.item()

        try:
            if (i+1) % (len(train_loader) // 4) == 0:
                print(f"[Epoch {epoch}/{total_epoch}] [Batch {i}/{len(train_loader)}] [loss: {loss/(inputs.shape[0])}]")
        except:
            if (i+1) % (len(train_loader) // 2) == 0:
                print(f"[Epoch {epoch}/{total_epoch}] [Batch {i}/{len(train_loader)}] [loss: {loss/(inputs.shape[0])}]")
                
    train_loss /= len(train_loader)
    valid_loss = evaluate(model,valid_loader,criterion)
    if lr_scheduler:
        lr_scheduler.step(valid_loss)
    print(f"\nEpoch:{epoch}\nTrain Loss:{train_loss} |Valid Loss:{valid_loss}")
    wandb.log({"Train Loss": train_loss,
               "Valid Loss": valid_loss,
               "Epoch": epoch,
               })
    
    if valid_loss <= best_v_loss:
        print(f"{g_}VALID Loss Decreased from {best_v_loss} to {valid_loss}{sr_}")
        best_v_loss = valid_loss
        torch.save(model.state_dict(),os.path.join(save_path,f'{exp_date}_BestValidLoss_informer.pt'))
    if train_loss <= best_t_loss:
        print(f"{g_}TRAIN Loss Decreased from {best_t_loss} to {train_loss}{sr_}")
        best_t_loss = train_loss
        torch.save(model.state_dict(),os.path.join(save_path,f'{exp_date}_BestTrainLoss_informer.pt'))
        
    return best_t_loss, best_v_loss

if __name__ == '__main__':
    # torch.set_default_dtype(torch.float32)
    #   seqlen, hop, load, load_config_exp
    config = [
        # [24*1, 6, False, ''],
        # [24*2, 6, False, ''],
        # [24*7, 6, False, ''],
        # [24*10, 6, False, ''],
        [24*14, 6, False, ''],
        # [24*20, 6, False, ''],
        # [24*26, 6, False, ''],
        # [24*30, 6, False, '']
    ]
    
    for win, h, load, config_exp in config:
        
        # run setting Date setting
        now = datetime.now()
        exp_date = now.strftime('%m%d-%S%M')
        
        # device Setting
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        
        # config
        # learing config
        epochs = 100
        batch_size = 64
        lr = 1e-4
        
        # data config
        mode = 'train'
        window = win
        hop = h
        enc_in = dec_in = 18    # 인코더, 디코더 입력 차원 수 (예: 피처 수 16)
        c_out = 1               # 출력 차원 수 (예: 예측할 온도 값의 차원 1)
        seq_len = label_len = out_len = win # 인코더, 디코더 입력 시퀀스 길이, 출력 시퀀스 길이
        
        ############## loading data ###################
        path = r'/workspace/MLProject/MLproject/data/train.csv'
        dataset = MLdataset(path, mode, window, hop)
        
        # 데이터셋 분할
        # train_ratio = 1
        # train_size = int(train_ratio * len(dataset))
        # valid_size = len(dataset) - train_size
        # train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])

        # DataLoader 생성
        train_dl = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        valid_dl = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            

        model = InformerStack(enc_in - 4,
                              dec_in - 4,
                              c_out,
                              seq_len,
                              label_len,
                              out_len,
                              d_model = 512,
                              n_heads=16,
                              e_layers=[4,1],
                              d_layers=2,
                              dropout=0.1
                              )
        
        # model = Informer(enc_in - 4,
        #                  dec_in - 4,
        #                  c_out,
        #                  seq_len,
        #                  label_len,
        #                  out_len,
        #                  d_model = 512,
        #                  n_heads=16,
        #                  e_layers=[4,1],
        #                  d_layers=2,
        #                  dropout=0.1)
        
        # load model
        if load:
            load_exp = config_exp
            load_path = os.path.join(os.getcwd(),'saved_Model',f'{project_name}-result',load_exp,f'{load_exp}_informer.pt')
            model.load_state_dict(torch.load(load_path))
            
        optimizer = optim.Adam(model.parameters(),lr=lr,amsgrad=False)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',patience=5)
        criterion = nn.MSELoss()
        
        accelerator = Accelerator()
        print(f"{accelerator.device} is used")
        model,train_dl,valid_dl,optimizer,lr_scheduler,criterion = accelerator.prepare(model,train_dl,valid_dl,optimizer,lr_scheduler,criterion)
        
        # WandB 초기화
        project_name = "ML-TeamProject87_ppt"

        # saved_Model dir Create
        save_path = os.path.join(os.getcwd(),'saved_Model')
        if os.path.isdir(save_path):
            print('Path alreay exist')
        else:
            os.mkdir(save_path)
        # Project dir Create
        save_path = os.path.join(save_path,f'{project_name}_result')
        if os.path.isdir(save_path):
            print('Path alreay exist')
        else:
            os.mkdir(save_path)
        # experiment dir create
        save_path = os.path.join(save_path, exp_date)
        if os.path.isdir(save_path):
            print('Path alreay exist')
        else:
            os.mkdir(save_path)


        run = wandb.init(project=project_name, name=f"exp_server_{exp_date}", reinit=True)
        
        config = wandb.config
        config.update({
            "model code": 'informerStack',
            "batch_size": batch_size,
            "learning_rate": lr,
            "optimizer": "Adam",
            "d_model": 512,
            "num head": 16,
            "input sequence len": win
        })
        
        best_v_loss = 9999999
        best_t_loss = 9999999
        start_time = time.time()
        for epoch in range(epochs):
            print(f"\n{project_name}\nEpoch Started:{epoch}")
            best_t_loss, best_v_loss = train_and_evaluate_loop(train_dl,valid_dl,model,optimizer,criterion,epoch,best_t_loss, best_v_loss,epochs,save_path,exp_date,lr_scheduler)
            
            end_time = time.time()
            print(f"{m_}Time taken by epoch {epoch} is {end_time-start_time:.2f}s{sr_}")
            start_time = end_time
            # print("잘 돌아감")
            # exit()
        print(f'best valid loss : {best_v_loss}\nbest train loss {best_t_loss}')
        run.finish()