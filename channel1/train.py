#-*- coding: utf-8 -*-
from load_preprocess_yuns import * 
from model_yuns import *
import os
from sklearn.metrics import f1_score
import pandas as pd
from tqdm import tqdm
import random

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, StepLR, ReduceLROnPlateau
import torch.nn.functional as F
import timm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import warnings
warnings.simplefilter("ignore", UserWarning)

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

# 내 전용 directory 만들어야할 듯
def save_model(name, model, optimizer, scheduler, fold, best=False):
    state = {
        'model': model.state_dict(), # state_dict 는 간단히 말해 각 계층을 매개변수(가중치, 편향) 텐서로 매핑되는 Python 사전(dict) 객체
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }
    if not (os.path.isdir('./yuns_model')): os.mkdir('./yuns_model')
    if not (os.path.isdir('./yuns_model/{}'.format(name))): os.mkdir('./yuns_model/{}'.format(name))
    torch.save(state, './yuns_model/{}/fold{}.pt'.format(name, fold+1))
    if best == True:
        if not (os.path.isdir('./yuns_best_model')): os.mkdir('./yuns_best_model')
        if not (os.path.isdir('./yuns_best_model/{}'.format(name))): os.mkdir('./yuns_best_model/{}'.format(name))
        torch.save(state, './yuns_best_model/{}/fold{}.pt'.format(name, fold+1))

class EarlyStopping: 
    def __init__(self, patience, criterion='loss'): # loss 또는 f1_score / acc 기준
        self.patience = patience
        self.criterion = criterion
        self.counter = 0
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.val_f1_max = -np.Inf

    def __call__(self, name, score, model, optimizer, scheduler, fold): # score는 validation의 loss값 또는 f1_score/acc 값

        if self.criterion == 'loss':
            val_loss = score
            if self.val_loss_min == np.Inf:
                self.val_loss_min = val_loss
                save_model(name, model, optimizer, scheduler, fold, best=True)
                print('*** Validation loss decreased (np.Inf --> {}).  Saving model... ***'.format(round(val_loss, 6)))
            elif val_loss > self.val_loss_min:
                self.counter += 1
                print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
                if self.counter >= self.patience:
                    print('Early Stopping - Fold {} Training is Stopping'.format(fold))
                    self.early_stop = True
            else:  # val_loss < val_loss_min
                save_model(name, model, optimizer, scheduler, fold, best=True)
                print('*** Validation loss decreased ({} --> {}).  Saving model... ***'.\
                      format(round(self.val_loss_min, 6), round(val_loss, 6)))
                self.val_loss_min = val_loss
                self.counter = 0

        else: # f1_score / acc
            f1_score = score
            if self.val_f1_max == -np.Inf:
                self.val_f1_max = f1_score
                save_model(name, model, optimizer, scheduler, fold, best=True)
                print('*** Validation f1_score increased (-np.Inf --> {}).  Saving model... ***'.format(round(f1_score, 6)))
            elif f1_score < self.val_f1_max:
                self.counter += 1
                print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
                if self.counter >= self.patience:
                    print('Early Stopping - Fold {} Training is Stopping'.format(fold))
                    self.early_stop = True
            else:  # f1_score > val_f1_max
                save_model(name, model, optimizer, scheduler, fold, best=True)
                print('*** Validation f1_score increased ({} --> {}).  Saving model... ***'.\
                      format(round(self.val_f1_max, 6), round(f1_score, 6)))
                self.val_f1_max = f1_score
                self.counter = 0

# https://gist.github.com/SuperShinyEyes/dcc68a08ff8b615442e3bc6a9b55a354
# https://stackoverflow.com/questions/53354176/how-to-use-f-score-as-error-function-to-train-neural-networks
class F1_Loss(nn.Module):
    def __init__(self, epsilon=1e-7):
        super().__init__()
        self.epsilon = epsilon

    def forward(self,y_pred,y_true,num_classes=5):
        y_true = F.one_hot(y_true,num_classes).to(torch.float32)
        y_pred = F.softmax(y_pred, dim=1)

        tp = (y_true*y_pred).sum(dim=1).to(torch.float32)
        tn = ((1-y_true)*(1-y_pred)).sum(dim=1).to(torch.float)
        fp = ((1-y_true)*y_pred).sum(dim=1).to(torch.float32)
        fn = (y_true*(1-y_pred)).sum(dim=1).to(torch.float32)

        precision = tp / (tp+fp+self.epsilon)
        recall = tp / (tp+fn+self.epsilon)

        f1 = 2*(precision*recall) / (precision+recall+self.epsilon)
        # self.epsilon 보다 작은 값은 self.epsilon값으로, 1-self.epsilon 값보다 큰 값은 1-self.epsilon 값으로 반환 
        f1 = f1.clamp(min=self.epsilon, max=1-self.epsilon) 
        return 1 - f1.mean()


def train_one_epoch(epoch, model, loss_fn, optimizer, train_loader, device, apex=False):
    model.train()
    lst_out = [] ; lst_label = []
    running_loss = 0.00

    for step, (images, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
        images = images.to(device).float()
        labels = labels.to(device).long()

        optimizer.zero_grad() # 가중치 초기화
        preds = model(images)
        lst_out += [torch.argmax(preds, 1).cpu().numpy()]
        lst_label += [labels.cpu().numpy()]

        loss = loss_fn(preds, labels)
        if apex:
            with scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step() # 가중치 업데이트
        running_loss += loss.item() # .item()을 통해 tensor를 scalar로 EX. tensor(0.2245) → 0.22448286414146423

    epoch_loss = running_loss / len(train_loader)
    lst_out = np.concatenate(lst_out); lst_label = np.concatenate(lst_label)
    train_f1_score = f1_score(lst_out, lst_label, average='micro')
    print('{} epoch - train loss : {}, train f1_score : {}'.\
          format(epoch + 1, np.round(epoch_loss, 6), np.round(train_f1_score*100, 2)))
    return epoch_loss, train_f1_score


def valid_one_epoch(epoch, model, loss_fn, val_loader, device):
    model.eval()
    lst_val_out = [] ; lst_val_label = []
    validation_loss = 0

    with torch.no_grad():
        for step, (images, labels) in tqdm(enumerate(val_loader), total=len(val_loader)):
            val_images = images.to(device).float()
            val_labels = labels.to(device).long()

            val_preds = model(val_images)
            lst_val_out += [torch.argmax(val_preds, 1).cpu().numpy()]
            lst_val_label += [val_labels.cpu().numpy()]

            val_loss = loss_fn(val_preds, val_labels)
            validation_loss += val_loss.item() 

        epoch_val_loss = validation_loss / len(val_loader)
        lst_val_out = np.concatenate(lst_val_out); lst_val_label = np.concatenate(lst_val_label)
        val_f1_score = f1_score(lst_val_out, lst_val_label, average='micro')
        print('{} epoch - valid loss : {}, valid f1_score : {}'.\
              format(epoch + 1, np.round(epoch_val_loss, 6), np.round(val_f1_score*100,2)))
    return epoch_val_loss, val_f1_score


if __name__ == "__main__":
    batch_size = 128
    num_workers = 4
    seed = 42
    epochs = 200
    patience = 5
    save_name = '0206'
    
    trainval_dataset = pd.read_csv('../DATA/trainset-for_user.csv', header=None)
    patients = trainval_dataset[0].unique().tolist()
    seed_everything(seed)
    for fold  in range(5):
        # train:val = 8:2
        val_patients = patients[int(len(patients)*0.2*fold):int(len(patients)*0.2*(fold+1))]
        val_patients_data = trainval_dataset.loc[[True if trainval_dataset[0][i] in val_patients else False for i in range(trainval_dataset.shape[0])],:].reset_index(drop=True)
        train_patients_data = trainval_dataset.loc[[True if trainval_dataset[0][i] not in val_patients else False for i in range(trainval_dataset.shape[0])],:].reset_index(drop=True)
        print('train_dataset shape: {} / val_dataset shape: {}'.format(train_patients_data.shape, val_patients_data.shape))

        train_dataset = TrainValDataset(train_patients_data)
        val_dataset = TrainValDataset(val_patients_data)
        train_loader = data_loader('train', train_dataset, batch_size=batch_size, num_workers=num_workers)
        val_loader = data_loader('valid', val_dataset, batch_size=batch_size, num_workers=num_workers)

        print('---------- Fold {} is training ----------'.format(fold + 1))

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Model
        model = IITNET().to(device)

        criterion = nn.CrossEntropyLoss()
        # criterion = F1_Loss()
        optimizer = AdamW(model.parameters(), lr=0.005, weight_decay=1e-6)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=2) # val_f1_score를 기준으로
        early_stopping = EarlyStopping(patience=patience, criterion='f1_score')

        for epoch in range(epochs):
            # train
            train_loss, train_f1_score = train_one_epoch(epoch, model, criterion, optimizer, train_loader, device)
            save_model(save_name, model, optimizer, scheduler, fold)
            # validation
            val_loss, val_f1_score = valid_one_epoch(epoch, model, criterion, val_loader, device)
            early_stopping(save_name, val_f1_score, model, optimizer, scheduler, fold)
            if early_stopping.early_stop:
                break
            scheduler.step(val_f1_score)

        del model, optimizer, train_dataset, val_dataset, train_loader, val_loader, scheduler
        torch.cuda.empty_cache()