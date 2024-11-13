import os
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from dataset import MLdataset
from models.model import InformerStack
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
    total = 0.0
    valid_loss = 0
    true_labels = []
    pred_labels = []
    label_len = 0
    with torch.no_grad():
        for i, (data, labels) in enumerate(valid_loader):
            inputs_t = data[:,:,:4]
            inputs = data[:,:,4:]
            
            total += inputs.shape[0] * inputs.shape[1]

            output = model(inputs,inputs_t,inputs,inputs_t)
            
            labels = labels.to(output.device)
            label_len = labels.shape[1]
                
            loss = criterion(output, labels)
            valid_loss += loss.item()
            
                
            true_labels.append(labels.cpu().numpy())
            pred_labels.append(output.cpu().numpy())
                
    
    metrics = {}
    true = np.concatenate(true_labels, axis=0).reshape(-1,label_len)
    pred = np.concatenate(pred_labels, axis=0).reshape(-1,label_len)
    
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
    
    valid_loss /= total
    return valid_loss

def train_and_evaluate_loop(train_loader,valid_loader,model,optimizer,
                            criterion,epoch,best_t_loss, best_v_loss,
                            total_epoch,
                            save_path,exp_date,lr_scheduler=None):
    train_loss = 0
    total = 0.0
    model.train()
    for i, (data, label) in enumerate(train_loader):
        optimizer.zero_grad()
        
        inputs_t = data[:,:,:4]
        inputs = data[:,:,4:]
        
        total += inputs.shape[0] * inputs.shape[1]

        output = model(inputs,inputs_t,inputs,inputs_t)
        
        label = label.to(output.device)
            
        loss = criterion(output, label)

        accelerator.backward(loss) 
        optimizer.step()
        train_loss += loss.item()

        try:
            if (i+1) % (len(train_loader) // 4) == 0:
                print(f"[Epoch {epoch}/{total_epoch}] [Batch {i}/{len(train_loader)}] [loss: {loss/(inputs.shape[0] * inputs.shape[1])}]")
        except:
            if (i+1) % (len(train_loader) // 2) == 0:
                print(f"[Epoch {epoch}/{total_epoch}] [Batch {i}/{len(train_loader)}] [loss: {loss/(inputs.shape[0] * inputs.shape[1])}]")
                
    train_loss /= total
    valid_loss = evaluate(model,valid_loader,criterion)
    if lr_scheduler:
        lr_scheduler.step(valid_loss)
    print(f"\nEpoch:{epoch}\nTrain Loss:{train_loss} |Valid Loss:{valid_loss}")
    wandb.log({"Train Loss": train_loss,
               "Valid Loss": valid_loss,
               "Epoch": epoch,
               })
    
    if valid_loss <= best_v_loss:
        print(f"{g_}Loss Decreased from {best_v_loss} to {valid_loss}{sr_}")
        best_v_loss = valid_loss
        torch.save(model.state_dict(),os.path.join(save_path,f'{exp_date}_BestValidLoss_informer.pt'))
    if train_loss <= best_t_loss:
        print(f"{g_}Loss Decreased from {best_t_loss} to {train_loss}{sr_}")
        best_t_loss = train_loss
        torch.save(model.state_dict(),os.path.join(save_path,f'{exp_date}_BestTrainLoss_informer.pt'))
        
    return best_t_loss, best_v_loss

def get_class_counts(dataset, class_list=[5, 9, 5, 2]):
    """
    클래스별 레이블 개수를 계산합니다.

    Args:
        dataset (Dataset): PyTorch Dataset 객체.
        class_list (list): 각 레이블 그룹별 클래스 수. 예: [m_classes, g_classes, la_classes, ra_classes]

    Returns:
        dict: 각 레이블 그룹별 클래스별 레이블 개수를 담은 사전.
    """
    if len(class_list) > 1:
        m_classes, g_classes, la_classes, ra_classes = class_list
        
        # 클래스별 카운트 초기화
        m_counts = torch.zeros(m_classes)
        g_counts = torch.zeros(g_classes)
        la_counts = torch.zeros(la_classes)
        ra_counts = torch.zeros(ra_classes)

        # 데이터셋 순회
        for sample in dataset:
            # 레이블 추출
            m_label, g_label, la_label, ra_label = sample[4]
            
            # 텐서로 변환하여 카운트 누적
            m_counts += torch.tensor(m_label, dtype=torch.float32)
            g_counts += torch.tensor(g_label, dtype=torch.float32)
            la_counts += torch.tensor(la_label, dtype=torch.float32)
            ra_counts += torch.tensor(ra_label, dtype=torch.float32)

        return {
            'motor': m_counts,
            'gearbox': g_counts,
            'left_axlebox': la_counts,
            'right_axlebox': ra_counts
        }
    else:
        classes = class_list[0]
        class_counts = torch.zeros(classes)
        for sample in dataset:
            # 레이블 추출
            labels = sample[4][0]

            # 텐서로 변환하여 카운트 누적
            class_counts += torch.tensor(labels, dtype=torch.float32)
        return {
            'classes': class_counts
        }

def calculate_pos_weight(class_counts, total_samples):
    """
    각 클래스의 pos_weight를 계산합니다.

    Args:
        class_counts (dict): 클래스별 레이블 개수를 담은 사전.
        total_samples (int): 전체 샘플 수.

    Returns:
        dict: 클래스별 pos_weight를 담은 사전.
    """
    pos_weights = {}
    for i, (label_group, counts) in enumerate(class_counts.items()):
        # 음성 샘플 수 = 전체 샘플 수 - 양성 샘플 수
        neg_counts = total_samples[i] - counts
        pos_weight = neg_counts / counts
        pos_weights[label_group] = pos_weight
    # print(pos_weights)
    # exit()
    return pos_weights

if __name__ == '__main__':
    # torch.set_default_dtype(torch.float32)
    #   seqlen, hop, load, load_config_exp
    config = [
        # [24*1, 12, False, ''],
        # [24*2, 12, False, ''],
        # [24*3, 12, False, ''],
        # [24*4, 12, False, ''],
        # [24*5, 12, False, ''],
        # [24*6, 12, False, ''],
        # [24*7, 12, False, ''],
        # [24*10, 12, False, ''],
        [24*15, 12, False, ''],
        [24*20, 12, False, ''],
        # [24*25, 12, False, ''],
        # [24*30, 12, False, '']
    ]
    
    for win, h, load, config_exp in config:
        
        # run setting Date setting
        now = datetime.now()
        exp_date = now.strftime('%m%d-%S%M')
        
        # device Setting
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        
        # config
        # learing config
        epochs = 100
        batch_size = 32
        lr = 1e-2
        
        # data config
        mode = 'train'
        window = win
        hop = h
        enc_in = dec_in = 16    # 인코더, 디코더 입력 차원 수 (예: 피처 수 16)
        c_out = 1               # 출력 차원 수 (예: 예측할 온도 값의 차원 1)
        seq_len = label_len = out_len = win # 인코더, 디코더 입력 시퀀스 길이, 출력 시퀀스 길이
        
        ############## loading data ###################
        path = r'/workspace/MLProject/data/train.csv'
        dataset = MLdataset(path, mode, window, hop)
        
        # 데이터셋 분할
        train_ratio = 0.7
        train_size = int(train_ratio * len(dataset))
        # valid_size = len(dataset) - train_size
        # train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])
        train_dataset = dataset[:train_size]
        valid_dataset = dataset[train_size:]

        # DataLoader 생성
        train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        valid_dl = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
            

        model = InformerStack(enc_in - 4,
                              dec_in - 4,
                              c_out,
                              seq_len,
                              label_len,
                              out_len,
                              dropout=0.2
                              )
        
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
        project_name = "ML-TeamProject"

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
            "input sequence len": win,
            "input sequence hop": hop
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