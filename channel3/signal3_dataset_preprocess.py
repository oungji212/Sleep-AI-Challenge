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

def save_preprocessed_imgs(which_data='train'):
    if which_data == 'train':
        dataset = pd.read_csv('../../DATA/trainset-for_user.csv', header=None)
    else: # test
        dataset = pd.read_csv('../../DATA/testset-for_user.csv', header=None)

    patients_unique = dataset[0].unique().tolist()
    for patient in tqdm(patients_unique,desc='{} img preprocess'.format(which_data)):
        unique_patient_data = dataset.loc[[True if dataset[0][i] == patient else False for i in range(dataset.shape[0])],:].reset_index(drop=True)
        patient_img = [os.path.join('../../DATA/', unique_patient_data[0][i], unique_patient_data[1][i]) for i in range(unique_patient_data.shape[0])] 

        imgpath_together_ls = []
        for num in range(len(patient_img)):
            if num <= 3:
                imgpath_together_ls.append(patient_img[:num+1])
            else:
                imgpath_together_ls.append(patient_img[num-3:num+1])

        if not os.path.isdir('./yuns_signal3_dataset/{}'.format(patient)): # 환자별 폴더 만들기
            os.mkdir('./yuns_signal3_dataset/{}'.format(patient)) # os.makedirs의 경우 permission denied 발생

        for idx in range(len(patient_img)):
            img_paths = imgpath_together_ls[idx]
            if len(img_paths) <= 3:
                for no_img_num in range(4-len(img_paths)):
                    globals()['img{}'.format(no_img_num)] = np.zeros((75,480))
                for img_num in range(4-len(img_paths),4):
                    img_path = img_paths[img_num-(4-len(img_paths))]
                    img = np.array(Image.open(img_path).crop((0,32,480,107)))  # 실제 이미지의 경우 기본적으로 1차원!
                    globals()['img{}'.format(img_num)] = img 
            else:
                for i in range(4):
                    path = img_paths[i]
                    img = np.array(Image.open(path).crop((0,32,480,107))) # EOG 부분만 잘라내기
                    globals()['img{}'.format(img_num)] = img 

            img_concat = np.concatenate([img0,img1,img2,img3], axis=1)
            # print('img_concat shape:',img_concat.shape) # (21,1920)이 나와야함
            save_img = Image.fromarray(img_concat.astype(np.uint8))
            save_img.save('./yuns_signal3_dataset/{}/{}'.format(patient,unique_patient_data[1][idx])) # 기준 epoch 사진의 원본 이름으로 저장. 저장 위치만 다른 것임


if __name__ == '__main__':
    # USER에서 실행
    if not os.path.isdir('./yuns_signal3_dataset'): 
        os.mkdir('./yuns_signal3_dataset') # os.makedirs의 경우 permission denied 발생
    save_preprocessed_imgs(which_data='train')
    save_preprocessed_imgs(which_data='test')