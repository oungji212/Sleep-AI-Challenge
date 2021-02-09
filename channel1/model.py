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
import numpy as np
import torch.nn.functional as F
import torch.nn.init as init # initialize

# stage: 'conv_2'&'conv_3':[16, 16, 64] / 'conv_4'&'conv_5':[32, 32, 128] 
# conv_1: True: downsampling with stride 2 / False: nothing
class ResidualBlock(nn.Module):
	def __init__(self,stage,downsampling=False):
		super(ResidualBlock, self).__init__()
		self.shortcut = self.build_shortcut(stage,downsampling)
		self.conv_block = self.build_conv_block(stage,downsampling)

	def build_shortcut(self,stage,downsampling=False):
		if stage in ['conv_2','conv_3']:
			if downsampling == True:
				shortcut = [nn.Conv2d(64, 64, (1,2), stride=(1,2)),
							nn.BatchNorm2d(64)]
				return nn.Sequential(*shortcut)
			else:
				return 'identity'
		elif stage == 'conv_4':
			if downsampling == True:
				shortcut = [nn.Conv2d(64, 128, (1,2), stride=(1,2)),
							nn.BatchNorm2d(128)]
				return nn.Sequential(*shortcut)
			else:
				return 'identity'
		else: # 'conv_5'
			if downsampling == True:
				shortcut = [nn.Conv2d(128, 128, (1,2), stride=(1,2)),
							nn.BatchNorm2d(128)]
				return nn.Sequential(*shortcut)
			else:
				return 'identity'


	def build_conv_block(self,stage,downsampling):
		conv_block = []

		if stage in ['conv_2','conv_3']:
			if downsampling == True: # github 구현한거에선 kernel_size=1, stride=2로 하여 띄엄띄엄 downsampling
				conv_block += [nn.Conv2d(64, 16, (1,2), stride=(1,2)),
							   nn.BatchNorm2d(16),
							   nn.ReLU(inplace=True)]
			else: # conv_1 == False
				conv_block += [nn.Conv2d(64, 16, (1,1), stride=(1,1)),
							   nn.BatchNorm2d(16),
							   nn.ReLU(inplace=True)]

			conv_block += [nn.Conv2d(16, 16, (1,3), stride=(1,1), padding=(0,1)),
						   nn.BatchNorm2d(16),
						   nn.ReLU(inplace=True),

						   nn.Conv2d(16, 64, (1,1), stride=(1,1)),
						   nn.BatchNorm2d(64)]

		elif stage == 'conv_4':
			if downsampling == True:
				conv_block += [nn.Conv2d(64, 32, (1,2), stride=(1,2)),
							   nn.BatchNorm2d(32),
							   nn.ReLU(inplace=True)]
			else:
				conv_block += [nn.Conv2d(128, 32, (1,1), stride=(1,1)),
							   nn.BatchNorm2d(32),
							   nn.ReLU(inplace=True)]

			conv_block += [nn.Conv2d(32, 32, (1,3), stride=(1,1), padding=(0,1)),
						   nn.BatchNorm2d(32),
						   nn.ReLU(inplace=True),

						   nn.Conv2d(32, 128, (1,1), stride=(1,1)),
						   nn.BatchNorm2d(128)]

		else: # conv_5
			if downsampling == True: # github 구현한거에선 kernel_size=1, stride=2로 하여 띄엄띄엄 downsampling
				conv_block += [nn.Conv2d(128, 32, (1,2), stride=(1,2)),
							   nn.BatchNorm2d(32),
							   nn.ReLU(inplace=True)]
			else: # 
				conv_block += [nn.Conv2d(128, 32, (1,1), stride=(1,1)),
						  	   nn.BatchNorm2d(32),
						   	   nn.ReLU(inplace=True)]

			conv_block += [nn.Conv2d(32, 32, (1,3), stride=(1,1), padding=(0,1)),
						   nn.BatchNorm2d(32),
						   nn.ReLU(inplace=True),

						   nn.Conv2d(32, 128, (1,1), stride=(1,1)),
						   nn.BatchNorm2d(128)]

		return nn.Sequential(*conv_block)

	def forward(self,x):
		if self.shortcut == 'identity':
			out = x + self.conv_block(x)
		else:
			out = self.shortcut(x) + self.conv_block(x)
		out = F.relu(out)
		return out


# sub-epoch인 5인 경우 (B,1,21,400)가 해당 모델의 기본 input
class modified_ResNet50(nn.Module): # EOG에 한해서 
	def __init__(self):
		super(modified_ResNet50,self).__init__()

		# stage1
		model = [nn.Conv2d(1, 64, (21, 3), stride=(1,1), padding=(0,1)), # 64, 1, 400
			nn.BatchNorm2d(num_features=64),
			nn.ReLU(inplace=True),
			# nn.MaxPool2d(kernel_size=(1,3), stride=(1,2), padding=(0,1)) 
			]

		# stage2
		model += [ResidualBlock('conv_2',downsampling=True)] # 64, 1, 200
		for i in range(2):
			model += [ResidualBlock('conv_2',downsampling=False)] # 64, 1, 200

		# stage3
		model += [ResidualBlock('conv_3',downsampling=True)] # 64, 1, 100
		for i in range(3):
			model += [ResidualBlock('conv_3',downsampling=False)] # 64, 1, 100

		# feature를 줄이기위헤 conv_3와 conv_4사이에 maxpooling -> 우리가 필요할까..
		# model += [nn.MaxPool2d(kernel_size=(1,3), stride=(1,2))]

		# stage4
		model += [ResidualBlock('conv_4',downsampling=True)] # 128, 1, 50
		for i in range(5):
			model += [ResidualBlock('conv_4',downsampling=False)] # 128, 1, 50

		# stage5
		model += [ResidualBlock('conv_5',downsampling=True)] # 128, 1, 25
		for i in range(2):
			model += [ResidualBlock('conv_5',downsampling=False)] # 128, 1, 25

		# dropout
		model += [nn.Dropout(0.5)]
		self.model = nn.Sequential(*model)

	def forward(self,x):
		out = self.model(x)
		return out


class BiLSTM(nn.Module):
	def __init__(self):
		super(BiLSTM,self).__init__()
		self.lstm1 = nn.LSTM(input_size=128, hidden_size=128, bidirectional=True, batch_first=True)
		self.lstm2 = nn.LSTM(input_size=128* 2, hidden_size=128, bidirectional=True, batch_first=True)

	def forward(self,x):
		output_seq1, _ = self.lstm1(x) # B, T, D
		output_seq2, _ = self.lstm2(output_seq1) # B, T, D
		last_output = output_seq2[:,-1,:] # for many-to-one: Decode the hidden state of the last time step 
		return last_output

class IITNET(nn.Module):
	def __init__(self, num_classes=5):
		super(IITNET,self).__init__()
		self.subepoch_length = 400
		self.overlapped_subepoch = 20
		self.n_subepochs = 5

		self.modified_ResNet50 = modified_ResNet50()
		self.BiLSTM = BiLSTM()
		# classifier
		self.classifier = nn.Linear(128*2, num_classes) # lstm hidden_size*2 * num_classes

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				init.kaiming_normal_(m.weight.data) # ReLU일 경우, He 초기화가 더 좋음 
				m.bias.data.fill_(0)

		for m in self.modules():
			if isinstance(m, nn.Linear):
				init.xavier_normal_(m.weight.data) # softmax이니까 xavier로
				m.bias.data.fill_(0)

	def forward(self,img_batch):
		feature_seq = []
		for i in range(self.n_subepochs):
			resnet_input = img_batch[:,:,:,(380*i):(400+380*i)] #torch: B, C, H, W
			resnet_output = self.modified_ResNet50(resnet_input) # B, 128, 1, 25
			squeeze_output = resnet_output.squeeze() # B, 128, 25 (batch, channel(feature, 차원D), seq_length(길이T))
			lstm_input = squeeze_output.transpose(1,2) # B T D 순서로 변경
			feature_seq.append(lstm_input)
		feature_seqs = torch.cat(feature_seq, dim=1) # T를 옆으로 붙이는 방식 (125)
		last_output = self.BiLSTM(feature_seqs)
		last_output = last_output.view(img_batch.size()[0],-1) # Linear을 위해 B*1차원으로 풀기 (128*2)
		final_output = self.classifier(last_output)
		return final_output

