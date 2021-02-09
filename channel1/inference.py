#-*- coding: utf-8 -*-
from load_preprocess_yuns import * 
from model_yuns import *
import os
import pandas as pd
from tqdm import tqdm

import random
import torch
import torch.nn as nn
import timm
import warnings
warnings.simplefilter("ignore", UserWarning)

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def inference(model, data_loader, device):
    lst_preds_out = []
    for step, (images) in tqdm(enumerate(data_loader), total=len(data_loader)):
        images = images.to(device).float()
        test_preds = model(images)
        lst_preds_out += [test_preds.cpu().numpy()]
    lst_preds_out = np.concatenate(lst_preds_out, axis=0) # label length * class_nums
    return lst_preds_out

def write_file(name, preds):
    dic_label = {0: 'Wake', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'REM'}
    submission = pd.DataFrame(preds)
    submission = submission.applymap(lambda x: dic_label[x]) # 본래 라벨명으로
    if not (os.path.isdir('./yuns_submission')): os.mkdir('./yuns_submission')
    if not (os.path.isdir('./yuns_submission/{}'.format(name))): os.mkdir('./yuns_submission/{}'.format(name))
    submission.to_csv('./yuns_submission/{}/submission.csv'.format(name), encoding='utf-8-sig', index=False, header=False)
    print('file rows : ', submission.shape[0])
    print('submission file is saved !')

if __name__ == "__main__":
    batch_size = 128
    num_workers = 4
    seed = 42
    name = '0205' 

    total_preds_out = []
    seed_everything(seed)
    test_dataset = TestDataset()
    test_loader = data_loader('test', test_dataset, batch_size=batch_size, num_workers=num_workers)
    for fold in range(5):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print('---------- Fold {} is Inferring ----------'.format(fold+1))
        model = IITNET().to(device)
        weights_path = './yuns_best_model/{}/fold{}.pt'.format(name, fold+1)
        model.load_state_dict(torch.load(weights_path)['model'])
        model.eval()
        with torch.no_grad():
            preds = inference(model, test_loader, device)
        total_preds_out.append(preds)

        del model, preds
        torch.cuda.empty_cache()

    mean_preds = np.mean(total_preds_out, axis=0) # 5개의 fold별 label length * class_nums 프레임을 평균
    np.save('channel1_mean_pred.npy', mean_preds)
    print(mean_preds.shape)
    # 단독버전
    test_preds = np.argmax(mean_preds, axis=1)
    write_file(name, test_preds)
