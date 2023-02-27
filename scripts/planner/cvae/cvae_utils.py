import numpy as np
import torch
import torch.nn as nn 
import torch.nn.functional as F

# Reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.set_default_dtype(torch.float32)

# GPU Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# PointNet architecture
class PointNet(nn.Module):
	def __init__(self, inp_channel=1, emb_dims=512, output_channels=20):
		super(PointNet, self).__init__()
		self.conv1 = nn.Conv1d(inp_channel, 64, kernel_size=1, bias=False) # input_channel = 3
		self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
		self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
		self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
		self.conv5 = nn.Conv1d(128, emb_dims, kernel_size=1, bias=False)
		self.bn1 = nn.BatchNorm1d(64)
		self.bn2 = nn.BatchNorm1d(64)
		self.bn3 = nn.BatchNorm1d(64)
		self.bn4 = nn.BatchNorm1d(128)
		self.bn5 = nn.BatchNorm1d(emb_dims)
		self.linear1 = nn.Linear(emb_dims, 256, bias=False)
		self.bn6 = nn.BatchNorm1d(256)
		self.dp1 = nn.Dropout()
		self.linear2 = nn.Linear(256, output_channels)
	
	def forward(self, x):
		x = F.relu(self.bn1(self.conv1(x)))
		x = F.relu(self.bn2(self.conv2(x)))
		x = F.relu(self.bn3(self.conv3(x)))
		x = F.relu(self.bn4(self.conv4(x)))
		x = F.relu(self.bn5(self.conv5(x)))
		x = F.adaptive_max_pool1d(x, 1).squeeze()
		x = F.relu(self.bn6(self.linear1(x)))
		x = self.dp1(x)
		x = self.linear2(x)
		return x

# Prevents NaN by torch.log(0)
def torch_log(x):
	return torch.log(torch.clamp(x, min = 1e-10))

# Encoder
class Encoder(nn.Module):
	def __init__(self, inp_dim, out_dim, hidden_dim, z_dim):
		super(Encoder, self).__init__()
				
		# Encoder Architecture
		self.encoder = nn.Sequential(
			nn.Linear(inp_dim + out_dim, hidden_dim),
			nn.BatchNorm1d(hidden_dim),
			nn.ReLU(),
			
			nn.Linear(hidden_dim, hidden_dim),
			nn.BatchNorm1d(hidden_dim),
			nn.ReLU(), 
			
			nn.Linear(hidden_dim, hidden_dim),
			nn.BatchNorm1d(hidden_dim),
			nn.ReLU(),
			
			nn.Linear(hidden_dim, hidden_dim),
			nn.BatchNorm1d(hidden_dim),
			nn.ReLU(),
			
			nn.Linear(hidden_dim, 256),
			nn.BatchNorm1d(256),
			nn.ReLU()
		)
		
		# Mean and Variance
		self.mu = nn.Linear(256, z_dim)
		self.var = nn.Linear(256, z_dim)
		
		self.softplus = nn.Softplus()
		
	def forward(self, x):
		out = self.encoder(x)
		mu = self.mu(out)
		var = self.var(out)
		return mu, self.softplus(var)
	
# Decoder
class Decoder(nn.Module):
	def __init__(self, inp_dim, out_dim, hidden_dim, z_dim):
		super(Decoder, self).__init__()
		
		# Decoder Architecture
		self.decoder = nn.Sequential(
			nn.Linear(z_dim + inp_dim, hidden_dim),
			nn.BatchNorm1d(hidden_dim),
			nn.ReLU(),
			
			nn.Linear(hidden_dim, hidden_dim),
			nn.BatchNorm1d(hidden_dim),
			nn.ReLU(),
			
			nn.Linear(hidden_dim, hidden_dim),
			nn.BatchNorm1d(hidden_dim),
			nn.ReLU(),
			
			nn.Linear(hidden_dim, hidden_dim),
			nn.BatchNorm1d(hidden_dim),
			nn.ReLU(),
			
			nn.Linear(hidden_dim, 256),
			nn.BatchNorm1d(256),
			nn.ReLU(),
			
			nn.Linear(256, out_dim),
		)
	
	def forward(self, x):
		out = self.decoder(x)
		return out

class Beta_cVAE(nn.Module):
	def __init__(self, encoder, decoder, pointnet):
		super(Beta_cVAE, self).__init__()
		
		# Encoder & Decoder
		self.encoder = encoder
		self.decoder = decoder
  
		# Pointnet
		self.pointnet = pointnet
				
  		# RCL Loss
		self.rcl_loss = nn.MSELoss()
	
	# Encoder: P_phi(z | X, y) where  X is both state & point cloud and y is ground truth
	def encode(self, state_norm, pcd, gt):
		
		# Feature Extractor PCD
		pcd_features = self.pointnet(pcd)

		# Inputs where X is vector comprised of state and pcd features and y is ground truth
		inputs = torch.cat([state_norm, pcd_features, gt], dim = 1)

		# Mean and  std of the latent distribution
		mean, std = self.encoder(inputs)        
  
		return mean, std

	# Reparametrization Trick
	def reparametrize(self, mean, std):
		eps = torch.randn_like(mean, device=device)
		return mean + std * eps

	# Decoder: P_theta(y | z, x) -> y* (init state, y) (No Diff opt)
	def decode(self, z, state, pcd):
	 
		# PCD feature extractor
		pcd_features = self.pointnet(pcd)
	 
		inputs = torch.cat([z, state, pcd_features], dim = 1)
		y = self.decoder(inputs)
			
		return y

	def loss_function(self, mean, std, y_star, y, beta = 1.0, step = 0):

		# Beta Annealing
		beta_d = min(step / 1000 * beta, beta)

		# KL Loss
		KL = -0.5 * torch.mean(torch.sum(1 + torch_log(std ** 2) - mean ** 2 - std ** 2, dim=1))
		
		# RCL Loss 
		RCL = self.rcl_loss(y_star, y)
								
		# ELBO Loss + Collision Cost
		loss = beta_d * KL + RCL 

		return KL, RCL, loss

	# Forward Pass
	def forward(self, state, pcd, coeff_gt, init_state_ego): # traj_gt

		# Normalize states & Scale Point cloud
		state_norm = (state - self.state_mean) / self.state_std
		scaled_pcd = (pcd - self.min_pcd) / (self.max_pcd - self.min_pcd)

		# Mu & Variance
		mean, std = self.encode(state_norm, scaled_pcd, coeff_gt)
				
		# Sample from z -> Reparameterized 
		z = self.reparametrize(mean, std)
		
		# Decode y
		y_star = self.decode(z, state_norm, scaled_pcd, init_state_ego)
	
		return mean, std, y_star

# Get the PointNet Weights
def extract_pointnet(z_dim, directory):

	# PointNet 
	pcd_features = 40
	pointnet = PointNet(inp_channel=2, emb_dims=1024, output_channels=pcd_features)

	# CVAE
	enc_inp_dim = 212 + pcd_features # State features 12 + Pcd features
	enc_out_dim = 400
	dec_inp_dim = enc_inp_dim
	dec_out_dim = 44
	hidden_dim = 1024 * 2
	z_dim = z_dim # change your shit


	# Load the Trained Model
	encoder = Encoder(enc_inp_dim, enc_out_dim, hidden_dim, z_dim)
	decoder = Decoder(dec_inp_dim, dec_out_dim, hidden_dim, z_dim)
	model = Beta_cVAE(encoder, decoder, pointnet)
	model.load_state_dict(torch.load(directory, map_location=torch.device('cpu')))
	model.eval()

	# Extract the weights & bias
	W0 = model.state_dict()['pointnet.conv1.weight'].detach().numpy()
	W1 = model.state_dict()['pointnet.conv2.weight'].detach().numpy()
	W2 = model.state_dict()['pointnet.conv3.weight'].detach().numpy()
	W3 = model.state_dict()['pointnet.conv4.weight'].detach().numpy()
	W4 = model.state_dict()['pointnet.conv5.weight'].detach().numpy()
	W5 = model.state_dict()['pointnet.linear1.weight'].detach().numpy()
	W6 = model.state_dict()['pointnet.linear2.weight'].detach().numpy()
	b6 = model.state_dict()['pointnet.linear2.bias'].detach().numpy()

	Wandb = [W0, W1, W2, W3, W4, W5, W6, b6]

	# Batch Norm Parameters
	scale_0 = model.state_dict()['pointnet.bn1.weight'].detach().numpy()
	bias_0 = model.state_dict()['pointnet.bn1.bias'].detach().numpy()
	mean_0 = model.state_dict()['pointnet.bn1.running_mean'].detach().numpy()
	var_0 = model.state_dict()['pointnet.bn1.running_var'].detach().numpy()

	scale_1 = model.state_dict()['pointnet.bn2.weight'].detach().numpy()
	bias_1 = model.state_dict()['pointnet.bn2.bias'].detach().numpy()
	mean_1 = model.state_dict()['pointnet.bn2.running_mean'].detach().numpy()
	var_1 = model.state_dict()['pointnet.bn2.running_var'].detach().numpy()

	scale_2 = model.state_dict()['pointnet.bn3.weight'].detach().numpy()
	bias_2 = model.state_dict()['pointnet.bn3.bias'].detach().numpy()
	mean_2 = model.state_dict()['pointnet.bn3.running_mean'].detach().numpy()
	var_2 = model.state_dict()['pointnet.bn3.running_var'].detach().numpy()

	scale_3 = model.state_dict()['pointnet.bn4.weight'].detach().numpy()
	bias_3 = model.state_dict()['pointnet.bn4.bias'].detach().numpy()
	mean_3 = model.state_dict()['pointnet.bn4.running_mean'].detach().numpy()
	var_3 = model.state_dict()['pointnet.bn4.running_var'].detach().numpy()

	scale_4 = model.state_dict()['pointnet.bn5.weight'].detach().numpy()
	bias_4 = model.state_dict()['pointnet.bn5.bias'].detach().numpy()
	mean_4 = model.state_dict()['pointnet.bn5.running_mean'].detach().numpy()
	var_4 = model.state_dict()['pointnet.bn5.running_var'].detach().numpy()

	scale_5 = model.state_dict()['pointnet.bn6.weight'].detach().numpy()
	bias_5 = model.state_dict()['pointnet.bn6.bias'].detach().numpy()
	mean_5 = model.state_dict()['pointnet.bn6.running_mean'].detach().numpy()
	var_5 = model.state_dict()['pointnet.bn6.running_var'].detach().numpy()

	BN = [scale_0, bias_0, mean_0, var_0, 
		  scale_1, bias_1, mean_1, var_1, 
		  scale_2, bias_2, mean_2, var_2, 
		  scale_3, bias_3, mean_3, var_3, 
		  scale_4, bias_4, mean_4, var_4,
		  scale_5, bias_5, mean_5, var_5]

	return Wandb, BN

# Get the Decoder Weights
def extract_decoder(z_dim, directory):
   	# PointNet 
	pcd_features = 40
	pointnet = PointNet(inp_channel=2, emb_dims=1024, output_channels=pcd_features)

	# CVAE
	enc_inp_dim = 212 + pcd_features # State features 12 + Pcd features
	enc_out_dim = 400
	dec_inp_dim = enc_inp_dim
	dec_out_dim = 44
	hidden_dim = 1024 * 2
	z_dim = z_dim # change your shit


	# Load the Trained Model
	encoder = Encoder(enc_inp_dim, enc_out_dim, hidden_dim, z_dim)
	decoder = Decoder(dec_inp_dim, dec_out_dim, hidden_dim, z_dim)
	model = Beta_cVAE(encoder, decoder, pointnet)
	model.load_state_dict(torch.load(directory, map_location=torch.device('cpu')))
	model.eval() 

	# Extracting the weights & biases
	W0 = model.state_dict()['decoder.decoder.0.weight'].detach().numpy()
	b0 = model.state_dict()['decoder.decoder.0.bias'].detach().numpy()

	W1 = model.state_dict()['decoder.decoder.3.weight'].detach().numpy()
	b1 = model.state_dict()['decoder.decoder.3.bias'].detach().numpy()

	W2 = model.state_dict()['decoder.decoder.6.weight'].detach().numpy()
	b2 = model.state_dict()['decoder.decoder.6.bias'].detach().numpy()

	W3 = model.state_dict()['decoder.decoder.9.weight'].detach().numpy()
	b3 = model.state_dict()['decoder.decoder.9.bias'].detach().numpy()

	W4 = model.state_dict()['decoder.decoder.12.weight'].detach().numpy()
	b4 = model.state_dict()['decoder.decoder.12.bias'].detach().numpy()

	W5 = model.state_dict()['decoder.decoder.15.weight'].detach().numpy()
	b5 = model.state_dict()['decoder.decoder.15.bias'].detach().numpy()

	Wandb = [W0, b0, 
				W1, b1, 
				W2, b2, 
				W3, b3, 
				W4, b4, 
				W5, b5]

	# Batch Norm Parameters
	scale_0 = model.state_dict()['decoder.decoder.1.weight'].detach().numpy()
	bias_0 = model.state_dict()['decoder.decoder.1.bias'].detach().numpy()
	mean_0 = model.state_dict()['decoder.decoder.1.running_mean'].detach().numpy()
	var_0 = model.state_dict()['decoder.decoder.1.running_var'].detach().numpy()

	scale_1 = model.state_dict()['decoder.decoder.4.weight'].detach().numpy()
	bias_1 = model.state_dict()['decoder.decoder.4.bias'].detach().numpy()
	mean_1 = model.state_dict()['decoder.decoder.4.running_mean'].detach().numpy()
	var_1 = model.state_dict()['decoder.decoder.4.running_var'].detach().numpy()

	scale_2 = model.state_dict()['decoder.decoder.7.weight'].detach().numpy()
	bias_2 = model.state_dict()['decoder.decoder.7.bias'].detach().numpy()
	mean_2 = model.state_dict()['decoder.decoder.7.running_mean'].detach().numpy()
	var_2 = model.state_dict()['decoder.decoder.7.running_var'].detach().numpy()

	scale_3 = model.state_dict()['decoder.decoder.10.weight'].detach().numpy()
	bias_3 = model.state_dict()['decoder.decoder.10.bias'].detach().numpy()
	mean_3 = model.state_dict()['decoder.decoder.10.running_mean'].detach().numpy()
	var_3 = model.state_dict()['decoder.decoder.10.running_var'].detach().numpy()

	scale_4 = model.state_dict()['decoder.decoder.13.weight'].detach().numpy()
	bias_4 = model.state_dict()['decoder.decoder.13.bias'].detach().numpy()
	mean_4 = model.state_dict()['decoder.decoder.13.running_mean'].detach().numpy()
	var_4 = model.state_dict()['decoder.decoder.13.running_var'].detach().numpy()

	BN = [scale_0, bias_0, mean_0, var_0, 
			scale_1, bias_1, mean_1, var_1, 
			scale_2, bias_2, mean_2, var_2, 
			scale_3, bias_3, mean_3, var_3, 
			scale_4, bias_4, mean_4, var_4]

	return Wandb, BN