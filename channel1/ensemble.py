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

if __name__ == "__main__":
	 # 다른 모델의 결과와 앙상블 (+)
    sh_data = np.load('tf_efficientnet_b4_ns_probs.npy')
    yj_data = np.load('channel1_mean_pred.npy')
    print('sh_data: {}/ yj_data {}'.format(sh_data.shape, yj_data.shape))
    two_mean = np.mean([sh_data[0], yj_data], axis=0) # 차원을 맞쳐주기 위해
    two_pred = np.argmax(two_mean, axis=1)
    print(two_pred.shape)

    dic_label = {0: 'Wake', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'REM'}
    sub = pd.DataFrame(two_pred)
    sub = sub.applymap(lambda x: dic_label[x]) # 본래 라벨명으로\
    sub.to_csv('ensemble_sub1.csv'.format(name), encoding='utf-8-sig', index=False, header=False)
    