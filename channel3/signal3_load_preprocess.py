#-*- coding: utf-8 -*-
import csv
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import random
from multiprocessing import cpu_count
import torch
from PIL import Image
from tqdm import tqdm
import pandas as pd

def transform_image(image): # 데이터 전처리
    custom_transformer = transforms.Compose([
        transforms.ToTensor(),
    ])
    image_tr = custom_transformer(image)
    return image_tr


def convert_label(label): # 수면상태 라벨을 정수형으로 change
    if label == 'Wake':
        label = 0
    elif label == 'N1':
        label = 1
    elif label == 'N2':
        label = 2
    elif label == 'N3':
        label = 3
    elif label == 'REM':
        label = 4
    else:
        label = None

    return label

# 바꾼 후
class TrainValDataset(Dataset):
    def __init__(self, patient_dataset, root='./yuns_signal3_dataset'): # main.py에서 device설정해주므로 여기서는 pass
        self.root = root 
        self.patient_dataset = patient_dataset
        self.data = {} # 최종적으로 data dictionary안에 data['image'], data['label'] 존재

        # 이미 yuns_dataset안에 4개의 epoch씩 합친 이미지들을 preprocess 해놓았으므로 그걸 불러오기만 하면 됨
        lst_img = [os.path.join(self.root, patient_dataset[0][i], patient_dataset[1][i]) for i in range(patient_dataset.shape[0])] 
        lst_label = [convert_label(y) for y in patient_dataset.iloc[:, -1].tolist()]
        self.data['image'] = lst_img
        self.data['label'] = lst_label   
        print('TrainVal-image,label num: {}, {}'.format(len(lst_img),len(lst_label)))            

    def __getitem__(self, index):
        path = self.data['image'][index]
        img = Image.open(path) # 이미 EOG로 잘라진 이미지들이므로 더 이상의 전처리 필요 X
        img = np.expand_dims(img,axis=2) # (75,1920) → (75,1920,1) : HxWxC (numpy ver.) cf.tensor: CxHxW
        img = transform_image(img) # HxWxC → CxHxW
        label = self.data['label'][index]
        return img, label

    def __len__(self):
        return len(self.data['image'])


class TestDataset(Dataset):
    def __init__(self, root='./yuns_signal3_dataset'): # main.py에서 device설정해주므로 여기서는 pass
        self.root = root 
        self.data = {} # 최종적으로 data dictionary안에 data['image'], data['label'] 존재
        test_path = os.path.join('../../DATA/', 'testset-for_user.csv')
        test_data = pd.read_csv(test_path,header=None)

        lst_img = [os.path.join(self.root, test_data[0][i], test_data[1][i]) for i in range(test_data.shape[0])]
        self.data['image'] = lst_img          

    def __getitem__(self, index):
        path = self.data['image'][index]
        img = Image.open(path)
        img = np.expand_dims(img,axis=2) # (75,1920) → (75,1920,1) : HxWxC (numpy ver.) cf.tensor: CxHxW
        img = transform_image(img) # HxWxC → CxHxW
        return img

    def __len__(self):
        return len(self.data['image'])


# 바꾼 후
def data_loader(phase, dataset, batch_size=32, num_workers=1): # USER에서 코드를 돌린다는 가정하에 진행
    if phase == 'train':
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    else: # val, test
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    return dataloader