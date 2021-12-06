import os
import time
import argparse
import numpy as np
import scipy.io as sio
import h5py
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from config import Config
from losses import l1_registered_uncertainty_loss, l1_registered_loss, cpsnr
from model import PIUNET
from dataset import ProbaVDatasetTrain, ProbaVDatasetVal

parser = argparse.ArgumentParser()
parser.add_argument('--log_dir', default='log_dir', help='Tensorboard log directory')
parser.add_argument('--save_dir', default='Results', help='Trained model directory')
param = parser.parse_args()

model_time = time.strftime("%Y%m%d_%H%M")

# Import config
config = Config()

# Import datasets
train_dataset = ProbaVDatasetTrain(config)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=False, num_workers=config.workers)

if config.validate:
	val_dataset = ProbaVDatasetVal(config)
	val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, drop_last=False)

dataset_mu = torch.Tensor((train_dataset.mu,)).to(config.device)
dataset_sigma = torch.Tensor((train_dataset.sigma,)).to(config.device)

# Create model
model = PIUNET(config)
model.cuda()

print('No. params: %d' % (sum(p.numel() for p in model.parameters() if p.requires_grad),) )

# Prepare logging
log_writer = SummaryWriter(log_dir=os.path.join(param.log_dir,'model_'+model_time))

# Train
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=0.0)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [150000], gamma=0.2, last_epoch=-1)

tot_steps=0
max_psnr=0.0
for epoch in range(config.N_epoch):	
	for step, (x_lr, x_hr, mask) in enumerate(train_loader):
		
		model.train()
		optimizer.zero_grad()
		
		x_lr = torch.Tensor(x_lr).to(config.device)
		x_hr = torch.Tensor(x_hr).to(config.device)
		mask = torch.Tensor(mask).to(config.device)

		mu_sr, sigma_sr = model(x_lr)

		x_hr = (x_hr-dataset_mu)/dataset_sigma

		if epoch < 1:
			loss = l1_registered_loss(x_hr, mu_sr, mask, config.patch_size*3)
		else:
			loss = l1_registered_uncertainty_loss(x_hr, mu_sr, sigma_sr, mask, config.patch_size*3)
		loss.backward()

		nn.utils.clip_grad_norm_(model.parameters(), 15)

		optimizer.step()
		scheduler.step()

		if step%config.log_every_iter ==0:
			log_writer.add_scalar('train/loss', loss.cpu().detach().numpy(), tot_steps+step)
			psnr_train = cpsnr(x_hr, mu_sr, mask, config.patch_size*3)
			log_writer.add_scalar('train/cPSNR', psnr_train.cpu().detach().numpy(), tot_steps+step)


	tot_steps = tot_steps+step

	if config.validate:
		model.eval()
		with torch.no_grad():
			psnr_val=[]
			x_sr_all=[]
			x_hr_all=[]
			s_sr_all=[]
			for val_step, (x_lr, x_hr, mask) in enumerate(val_loader):
				x_lr = torch.Tensor(x_lr).to(config.device)
				x_hr = torch.Tensor(x_hr).to(config.device)
				mask = torch.Tensor(mask).to(config.device)
				
				mu_sr, sigma_sr = model(x_lr)
				mu_sr = mu_sr*dataset_sigma + dataset_mu
				sigma_sr = sigma_sr + torch.log(dataset_sigma)

				x_sr_all.append(mu_sr)
				x_hr_all.append(x_hr)
				s_sr_all.append(sigma_sr)
				psnr_val.append(cpsnr(x_hr, mu_sr, mask, 128*3).cpu().detach().numpy())
			log_writer.add_scalar('val/cPSNR', np.mean(psnr_val), tot_steps)

			x_sr = torch.cat(x_sr_all)
			x_hr = torch.cat(x_hr_all)
			s_sr = torch.cat(s_sr_all)

		if np.mean(psnr_val)>max_psnr:
			torch.save(model.state_dict(), os.path.join(param.save_dir,'model_weights_best_'+model_time+'.pt'))
			max_psnr = np.mean(psnr_val)

	if epoch%50==0:
		torch.save(model.state_dict(), os.path.join(param.save_dir,'model_weights_'+model_time+'_epoch_'+str(epoch)+'.pt'))

	torch.save(model.state_dict(), os.path.join(param.save_dir,'model_weights_'+model_time+'.pt'))
