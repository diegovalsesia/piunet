import torch
import torch.nn as nn
import torch.nn.functional as F


def to_mha(x):
	# (B,F,T,X,Y) to (T,BXY,F)
	return torch.reshape(x.permute(2,0,3,4,1),[x.shape[2],x.shape[0]*x.shape[3]*x.shape[4],x.shape[1]])
	

def from_mha(x, target_shape):
	# (T,BXY,F) to (B,F,T,X,Y)
	return torch.reshape(x, [target_shape[2],target_shape[0],target_shape[3],target_shape[4],target_shape[1]]).permute(1,4,0,2,3)


class TEFA(nn.Module):

	def __init__(self, config):

		super(TEFA, self).__init__()

		self.conv0 = nn.Conv3d(config.N_feat, config.N_feat, [1,3,3], padding=[0,1,1])
		self.norm00 = nn.BatchNorm3d(config.N_feat)
		self.tran0 = nn.MultiheadAttention(config.N_feat, config.N_heads, dropout=0.0, bias=True)
		self.norm0 = nn.BatchNorm3d(config.N_feat)
		self.conv1 = nn.Conv3d(config.N_feat, config.N_feat, [1,3,3], padding=[0,1,1])
		self.tran1 = nn.MultiheadAttention(config.N_feat, config.N_heads, dropout=0.0, bias=True)
		self.norm11 = nn.BatchNorm3d(config.N_feat)
		self.norm1 = nn.BatchNorm3d(config.N_feat)

		self.lin0 = nn.Linear(config.N_feat, config.N_feat//config.R_bneck)
		self.lin1 = nn.Linear(config.N_feat//config.R_bneck, config.N_feat)

	def forward(self, h):
		
		h0 = h+0.0

		h = self.conv0(h)
		h = self.norm00(h)
		h = F.leaky_relu(h)
		h = h + from_mha( self.tran0( to_mha(h), to_mha(h), to_mha(h) )[0], h.shape )
		h = self.norm0(h)
		h = F.leaky_relu(h)
		h = self.conv1(h)
		h = self.norm11(h)
		h = F.leaky_relu(h)
		h = h + from_mha( self.tran1( to_mha(h), to_mha(h), to_mha(h) )[0], h.shape )
		h = self.norm1(h)

		h_to_scale = h+0.0 # (B,F,T,X,Y)
	    
		h = torch.mean(h, dim=(2,3,4)) # (B,F)

		h = self.lin0(h)
		h = F.leaky_relu(h)
		h = self.lin1(h)
		h = F.sigmoid(h)

		h = h.unsqueeze(2).unsqueeze(3).unsqueeze(4)
		
		h = h_to_scale*h

		return h + h0


class TERN(nn.Module):

	def __init__(self, config):
		
		super(TERN, self).__init__()

		self.conv0 = nn.Conv3d(config.N_feat, config.N_feat, [1,3,3], padding=[0,1,1])
		self.norm00 = nn.BatchNorm3d(config.N_feat)
		self.tran0 = nn.MultiheadAttention(config.N_feat, config.N_heads, dropout=0.0, bias=True)
		self.norm0 = nn.BatchNorm3d(config.N_feat)
		self.conv1 = nn.Conv1d(config.N_feat, 5*5, [1], padding=0)
		
	def forward(self, h):
		
		h_in = h+0.0 # (B,F,T,X,Y)
		B,C,T,X,Y = h_in.shape

		h = self.conv0(h)
		h = self.norm00(h)
		h = F.leaky_relu(h)
		h = h + from_mha( self.tran0( to_mha(h), to_mha(h), to_mha(h) )[0], h.shape )
		h = self.norm0(h)
		h = F.leaky_relu(h) # (B,F,T,X,Y)
		h = torch.mean(h,[3,4]) # (B,F,T)
		h = self.conv1(h) # (B,25,T)
		#h = F.softmax(h, dim=1)
		w = torch.reshape(h.permute(0,2,1),[B*T,5,5]).unsqueeze(1).unsqueeze(2) # (BT,1,1,5,5)

		h = torch.reshape( h_in.permute(0,2,1,3,4), [1,B*T,C,X,Y] ) # (1,BT,F,X,Y)
		h = F.conv3d(h, w, groups=B*T, padding=[0,2,2]) # (1,BT,F,X,Y)
		h = torch.reshape(h, [B,T,C,X,Y]).permute(0,2,1,3,4) # (B,F,T,X,Y)

		return h



class PIUNET(nn.Module):
	
	def __init__(self, config):

		super(PIUNET, self).__init__()
		
		self.conv_in = nn.Conv3d(1, config.N_feat, [1,3,3], padding=[0,1,1])
		self.tran_in = nn.MultiheadAttention(config.N_feat, config.N_heads, dropout=0.0, bias=True)
		self.norm00 = nn.BatchNorm3d(config.N_feat)
		self.norm0 = nn.BatchNorm3d(config.N_feat)

		self.res = nn.ModuleList([TEFA(config) for i in range(config.N_tefa)])

		self.conv_mid = nn.Conv3d(config.N_feat, config.N_feat, [1,3,3], padding=[0,1,1])
		self.tran_mid = nn.MultiheadAttention(config.N_feat, config.N_heads, dropout=0.0, bias=True)		
		self.norm11 = nn.BatchNorm3d(config.N_feat)
		self.norm1 = nn.BatchNorm3d(config.N_feat)

		self.regnet = TERN(config)

		self.conv_d2s_mu = nn.Conv2d(config.N_feat, 9, [3,3], padding=[1,1])

		self.conv_d2s_sigma = nn.Conv2d(config.N_feat, config.N_feat, [3,3], padding=[1,1])	
		self.conv_out_sigma = nn.Conv2d(config.N_feat, 1, [1,1], padding=[0,0])
		self.norm_sigma = nn.BatchNorm2d(config.N_feat)

		self.up = nn.Upsample(scale_factor=3, mode='bilinear')


	def forward(self, x):
		
		x_in = x+0.0

		x = x.unsqueeze(1) # (B,1,T,X,Y)

		x = self.conv_in(x)
		x = self.norm00(x)
		x = F.leaky_relu(x)
		x = x + from_mha( self.tran_in( to_mha(x), to_mha(x), to_mha(x) )[0], x.shape )
		x = self.norm0(x)

		x_res = x+0.0

		for res_layer in self.res:
			x = res_layer(x)

		x = self.conv_mid(x)
		x = self.norm11(x)
		x = F.leaky_relu(x)
		x = x + from_mha( self.tran_mid( to_mha(x), to_mha(x), to_mha(x) )[0], x.shape )
		x = self.norm1(x)
		x = F.leaky_relu(x)

		x = x + x_res

		x = self.regnet(x)

		x = torch.mean(x, dim=2)

		x_mu = self.conv_d2s_mu(x)
		x_mu = F.pixel_shuffle(x_mu, 3)

		x_up = torch.mean(self.up(x_in),dim=1).unsqueeze(1)

		mu_sr = x_mu + x_up

		x_sigma = self.conv_d2s_sigma(self.up(x))
		x_sigma = self.norm_sigma(x_sigma)
		x_sigma = F.leaky_relu(x_sigma)
		sigma_sr = self.conv_out_sigma(x_sigma)

		return mu_sr, sigma_sr