from loader import data_loader
import csv
import torch
import os
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import timm
from sklearn.model_selection import StratifiedKFold, KFold
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from torch.cuda.amp import autocast, GradScaler

# Fold별로 model 저장하게끔 
def save_model(model, optimizer, scheduler, fold_n):
    model_cpu = model.to('cpu')
    state = {
        'model': model_cpu.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }
    if not (os.path.isdir('./saved_model')): os.mkdir('./saved_model')
    torch.save(state, './saved_model/saved_model_{}.pth'.format(fold_n))


class EarlyStopping:
    def __init__(self, patience=15, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else: # score >= self.best_score → val_loss_min 갱신
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print('Validation loss decreased ({} --> {}).  Saving model ...'.format(round(self.val_loss_min,6), round(val_loss,6)))
        
        if not (os.path.isdir('./saved_model')): os.mkdir('./saved_model')
        torch.save(model.state_dict(), './saved_model/checkpoint.pt') # 임시로 저장하게끔
        self.val_loss_min = val_loss


class Model(nn.Module):
    def __init__(self, model_name, pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        n_features = self.model.fc.in_features
        self.model.fc = nn.Linear(n_features, 5)

    def forward(self, x):
        x = self.model(x)
        return x


def Train_Test(): # K-Fold 포함 & Test에 대한 예측 포함

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('training is starting on {}...'.format(device))

    batch_size = 16
    num_workers = 1

    trainval_dataset = pd.read_csv('../DATA/trainset-for_user.csv',header=None)
    X_train = trainval_dataset.iloc[:,:-1] # path 부분
    Y_train = trainval_dataset.iloc[:,-1] # label부분

    test_dataset = TestDataset()
    test_loader = data_loader('test', test_dataset, batch_size=batch_size)

    cv = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

    for idx, (train_index, val_index) in enumerate(cv.split(X_train,Y_train)):
        train_x, train_y = X_train.iloc[train_index], Y_train[train_index]
        val_x, val_y = X_train.iloc[val_index], Y_train[val_index]

        train_dataset = TrainValDataset(train_x, train_y)
        val_dataset = TrainValDataset(val_x, val_y)

        train_loader = data_loader('train', train_dataset, batch_size=batch_size, num_workers=num_workers)
        val_loader = data_loader('val', val_dataset, batch_size=batch_size, num_workers=num_workers)

        model = Model(model_name='resnet50').to(device) # resnext50_32x4d
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            [param for param in model.parameters() if param.requires_grad],
            lr = 0.001, weight_decay=1e-4)
        # StepLR: 정해진 step_size마다 lr * gamma하여 lr 감소
        # scheduler = StepLR(optimizer, step_size = 5, gamma =0.5)
        # ReduceLROnPlateau: 오차가 더 이상 줄어들지 않거나 정확도가 더 이상 늘지않으면, lr 감소시킴
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

        early_stopping = EarlyStopping(patience=15, verbose=False)
        print('Fold - {}'.format(idx+1))

        epochs = 1
        min_val_loss = np.Inf
        n_epochs_stop = 15 # 몇 epoch까지 두고 볼 것인가?
        epochs_no_improve = 0 # loss가 개선되지 않은 몇번째 epoch인가 
        for epoch in range(epochs):
            model.train()
            lst_out = []
            lst_label = []
            avg_loss = 0

            for i, (image, label) in enumerate(train_loader):
                if i % 100 == 0:
                    print('{} epoch {}th batch is training...'.format(epoch+1,i))
                image = image.to(device).float()
                label = label.to(device).long()

                optimizer.zero_grad() # 가중치 초기화
                pred = model(image)
                lst_out += pred
                lst_label += label

                loss = loss_fn(pred, label)
                loss.backward() # gradient descent
                optimizer.step() # model 파라미터 update
                avg_loss += loss.item() / len(train_loader)

            # scheduler.step() # StepLR 사용할 때
            scheduler.step(float(loss[0])) # ReduceLROnPlateau 사용할 때
            print('{} epoch /  loss is {}'.format(epoch+1, loss))

            # f1_score
            f1 = f1_score(num_classes=5, average = 'macro')
            print(f1(torch.tensor(lst_out), torch.tensor(lst_label)))

            model.eval()
            avg_val_loss = 0
            with torch.no_grad():
                for image, label in val_loader:
                    x = Variable(image).cuda()
                    y_ = Variable(label).cuda()

                    output = model(x)
                    avg_val_loss += loss_fn(output,y_).item() / len(val_loader)

            early_stopping(avg_val_loss, model)
            if early_stopping.early_stop:
                print('EarlyStopping!')
                break

        # Testdata에 대한 예측 진행
        # 해당 K-Fold에서 최종적으로 저장한 모델 load
        model.load_state_dict(torch.load('./saved_model/checkpoint.pt'))
        model.to(device)
        globals()['testpred_{}'.format(idx)] = np.array([])
        for i, (image, _) in enumerate(test_loader):
            image = image.to(device).float()
            label = label.to(device).long()
            out = model(image)
            # 일단은 soft voting 기법으로 test predict
            pred_cls = torch.max(pred,1)[1].numpy() # tensor → numpy
            globals()['testpred_{}'.format(idx)] =  np.append(globals()['testpred_{}'.format(idx)], pred_cls)

    # testpred_0 ~ 4 (야매의 끝판왕... 나중에 고쳐야함..ㅎㅎ)
    dic_label = {0: 'Wake', 1: 'N1', 2: 'N2', 3: 'N3', 4:'REM'}
    testpreds = np.vstack((test_pred0, test_pred1, test_pred2, test_pred3, test_pred4))
    testpreds_T = testpreds.T
    final_predict = [dic_label[Counter(testpreds_T[i]).most_common(1)[0][0]] for i in range(testpreds_T.shape[0])]
    predict_df = pd.DataFrame(final_predict)
    predict_df.to_csv('./test_result.csv',index=False,header=False)



if __name__ == "__main__":
    Train_Test()
