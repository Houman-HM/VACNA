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

# HMNet
class Beta_cVAE(nn.Module):
	def __init__(self, num_batch, P, Pdot, encoder, decoder, pointnet, state_mean, state_std, min_pcd, max_pcd):
		super(Beta_cVAE, self).__init__()
		
		# Encoder & Decoder
		self.encoder = encoder
		self.decoder = decoder
  
		# Pointnet
		self.pointnet = pointnet
		
		# Normalizing Constants for States & Pointcloud
		self.state_mean = state_mean
		self.state_std = state_std
		self.min_pcd = min_pcd
		self.max_pcd = max_pcd

		# P Matrices
		self.P = P.to(device)
		self.Pdot = Pdot.to(device)

		# No. of Variables
		self.nvar = 11
		self.num_batch = num_batch

		# Equality Matrices
		self.A_eq_x = torch.vstack([self.P[0], self.Pdot[0], self.P[-1]])
		self.A_eq_y = torch.vstack([self.P[0], self.Pdot[0], self.P[-1]])
		self.A_projection = torch.eye(self.nvar, device=device) # Confused here
		
  		# RCL Loss
		self.rcl_loss = nn.MSELoss()

	# Inverse Matrices
	def compute_mat_inv(self):
		
		cost = torch.mm(self.A_projection.T, self.A_projection)
  				
		cost_mat_x = torch.vstack([torch.hstack([cost, self.A_eq_x.T]), 
								   torch.hstack([self.A_eq_x, torch.zeros((self.A_eq_x.shape[0], self.A_eq_x.shape[0]), device=device)])])
		
		cost_mat_y = torch.vstack([torch.hstack([cost, self.A_eq_y.T]), 
								   torch.hstack([self.A_eq_y, torch.zeros((self.A_eq_y.shape[0], self.A_eq_y.shape[0]), device=device)])])

		cost_mat_inv_x = torch.linalg.inv(cost_mat_x)
		cost_mat_inv_y = torch.linalg.inv(cost_mat_y)
		
		return cost_mat_inv_x, cost_mat_inv_y

	# Boundary Vectors
	def compute_boundary(self, initial_state_ego):
		
		# Drone boundaries
		drone_state = initial_state_ego[:, 0:4]
  
		x_init_drone = drone_state[:, 0].reshape(self.num_batch, 1)
		y_init_drone = drone_state[:, 1].reshape(self.num_batch, 1)

		vx_init_drone = drone_state[:, 2].reshape(self.num_batch, 1) 
		vy_init_drone = drone_state[:, 3].reshape(self.num_batch, 1) 

		# Vehicle Boundaries
		vehicle_state = initial_state_ego[:, 4:10]

		x_init_vehicle = vehicle_state[:, 0].reshape(self.num_batch, 1)
		y_init_vehicle = vehicle_state[:, 1].reshape(self.num_batch, 1)
  
		vx_init_vehicle = vehicle_state[:, 2].reshape(self.num_batch, 1)
		vy_init_vehicle = vehicle_state[:, 3].reshape(self.num_batch, 1)

		xf = vehicle_state[:, 4].reshape(self.num_batch, 1)
		yf = vehicle_state[:, 5].reshape(self.num_batch, 1)

		b_eq_x_drone = torch.hstack([x_init_drone, vx_init_drone, xf])
		b_eq_y_drone = torch.hstack([y_init_drone, vy_init_drone, yf])

		b_eq_x_vehicle = torch.hstack([x_init_vehicle, vx_init_vehicle, xf])
		b_eq_y_vehicle = torch.hstack([y_init_vehicle, vy_init_vehicle, yf]) 
	
		return b_eq_x_drone, b_eq_y_drone, b_eq_x_vehicle, b_eq_y_vehicle

	# Solve Function
	def solve(self, initial_state_ego, neural_output_batch):
		
		# Boundary conditions
		b_eq_x_drone, b_eq_y_drone, b_eq_x_vehicle, b_eq_y_vehicle = self.compute_boundary(initial_state_ego) 
  
		# Inverse Matrices
		cost_mat_inv_x, cost_mat_inv_y = self.compute_mat_inv()

		cx_bar_drone = neural_output_batch[:, 0 : self.nvar]
		cy_bar_drone = neural_output_batch[:, self.nvar : 2 * self.nvar]
  
		cx_bar_vehicle = neural_output_batch[:, 2 * self.nvar : 3 * self.nvar]
		cy_bar_vehicle = neural_output_batch[:, 3 * self.nvar : 4 * self.nvar]

		lincost_x_drone = - torch.mm(self.A_projection.T, cx_bar_drone.T).T
		lincost_y_drone = - torch.mm(self.A_projection.T, cy_bar_drone.T).T
		
		lincost_x_vehicle = - torch.mm(self.A_projection.T, cx_bar_vehicle.T).T
		lincost_y_vehicle = - torch.mm(self.A_projection.T, cy_bar_vehicle.T).T

		sol_x_drone = torch.mm(cost_mat_inv_x, torch.hstack([-lincost_x_drone, b_eq_x_drone]).T).T
		sol_y_drone = torch.mm(cost_mat_inv_y, torch.hstack([-lincost_y_drone, b_eq_y_drone]).T).T

		sol_x_vehicle = torch.mm(cost_mat_inv_x, torch.hstack([-lincost_x_vehicle, b_eq_x_vehicle]).T).T
		sol_y_vehicle = torch.mm(cost_mat_inv_y, torch.hstack([-lincost_y_vehicle, b_eq_y_vehicle]).T).T
  
		primal_sol_x_drone = sol_x_drone[:,0:self.nvar]
		primal_sol_y_drone = sol_y_drone[:,0:self.nvar]

		primal_sol_x_vehicle = sol_x_vehicle[:,0:self.nvar]
		primal_sol_y_vehicle = sol_y_vehicle[:,0:self.nvar]

		# Solution
		y_star = torch.hstack([primal_sol_x_drone, primal_sol_y_drone,
                         	   primal_sol_x_vehicle, primal_sol_y_vehicle])

		return y_star

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

	# Decoder: P_theta(y | z, x) -> y* (init state, y)
	def decode(self, z, state, pcd, init_state_ego):
	 
		# PCD feature extractor
		pcd_features = self.pointnet(pcd)
	 
		inputs = torch.cat([z, state, pcd_features], dim = 1)
		y = self.decoder(inputs)
		
		# Call Optimization Solver 
		y_star = self.solve(init_state_ego, y)
  		
		return y_star

	def loss_function(self, mean, std, traj_star, traj_gt, beta = 1.0, step = 0):

		# Beta Annealing
		beta_d = min(step / 1000 * beta, beta)

		# KL Loss
		KL = -0.5 * torch.mean(torch.sum(1 + torch_log(std ** 2) - mean ** 2 - std ** 2, dim=1))
		
		# RCL Loss 
		RCL = self.rcl_loss(traj_star, traj_gt)
								
		# ELBO Loss + Collision Cost
		loss = beta_d * KL + RCL 

		return KL, RCL, loss

	# Forward Pass
	def forward(self, state, pcd, traj_gt, init_state_ego): # traj_gt

		# Normalize states & Scale Point cloud
		state_norm = (state - self.state_mean) / self.state_std
		scaled_pcd = (pcd - self.min_pcd) / (self.max_pcd - self.min_pcd)

		# Mu & Variance
		mean, std = self.encode(state_norm, scaled_pcd, traj_gt)
				
		# Sample from z -> Reparameterized 
		z = self.reparametrize(mean, std)
		
		# Decode y
		y_star = self.decode(z, state_norm, scaled_pcd, init_state_ego)
	
		return mean, std, y_star