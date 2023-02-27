import jax
import jax.numpy as jnp 
import flax.linen as nn
import equinox  
from functools import partial
from jax import jit

# QP Node
class CVAE():
    def __init__(self, num_batch, P, Pdot, Pddot, 
                 pointnet_Wandb, pointnet_BN, decoder_Wandb, decoder_BN,
                 state_mean, state_std, min_pcd, max_pcd, zdim=12):
        super(CVAE, self).__init__()

        # Weights
        self.Wandb_pn = pointnet_Wandb
        self.BN_pn = pointnet_BN
        self.Wandb_dec = decoder_Wandb
        self.BN_dec = decoder_BN

        # P Matrices
        self.P = P
        self.Pdot = Pdot
        self.Pddot = Pddot

        # No. of Variables
        self.nvar = 11
        self.num_batch = num_batch

        # Decoder Parameters
        self.zdim = zdim
        self.key = jax.random.PRNGKey(0)
        self.state_mean = state_mean
        self.state_std = state_std
        self.min_pcd = min_pcd
        self.max_pcd = max_pcd

        # Equality Matrices
        self.A_eq_x = jnp.vstack([self.P[0], self.Pdot[0], self.P[-1]])
        self.A_eq_y = jnp.vstack([self.P[0], self.Pdot[0], self.P[-1]])
        self.A_projection = jnp.eye(self.nvar)

    @partial(jit, static_argnums=(0, ))
    # Inverse Matrices
    def compute_mat_inv(self):
        cost = jnp.dot(self.A_projection.T, self.A_projection)

        cost_mat_x = jnp.vstack([jnp.hstack([cost, self.A_eq_x.T]), 
								 jnp.hstack([self.A_eq_x, jnp.zeros((self.A_eq_x.shape[0], self.A_eq_x.shape[0]))])])
		
        cost_mat_y = jnp.vstack([jnp.hstack([cost, self.A_eq_y.T]), 
                                 jnp.hstack([self.A_eq_y, jnp.zeros((self.A_eq_y.shape[0], self.A_eq_y.shape[0]))])])

        cost_mat_inv_x = jnp.linalg.inv(cost_mat_x)
        cost_mat_inv_y = jnp.linalg.inv(cost_mat_y)

        return cost_mat_inv_x, cost_mat_inv_y

    @partial(jit, static_argnums=(0, ))
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
        
        b_eq_x_drone = jnp.hstack([x_init_drone, vx_init_drone, xf])
        b_eq_y_drone = jnp.hstack([y_init_drone, vy_init_drone, yf])		
        
        b_eq_x_vehicle = jnp.hstack([x_init_vehicle, vx_init_vehicle, xf])
        b_eq_y_vehicle = jnp.hstack([y_init_vehicle, vy_init_vehicle, yf]) 

        return b_eq_x_drone, b_eq_y_drone, b_eq_x_vehicle, b_eq_y_vehicle

    @partial(jit, static_argnums=(0, ))
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

        lincost_x_drone = - jnp.dot(self.A_projection.T, cx_bar_drone.T).T
        lincost_y_drone = - jnp.dot(self.A_projection.T, cy_bar_drone.T).T
        
        lincost_x_vehicle = - jnp.dot(self.A_projection.T, cx_bar_vehicle.T).T
        lincost_y_vehicle = - jnp.dot(self.A_projection.T, cy_bar_vehicle.T).T

        sol_x_drone = jnp.dot(cost_mat_inv_x, jnp.hstack([-lincost_x_drone, b_eq_x_drone]).T).T
        sol_y_drone = jnp.dot(cost_mat_inv_y, jnp.hstack([-lincost_y_drone, b_eq_y_drone]).T).T

        sol_x_vehicle = jnp.dot(cost_mat_inv_x, jnp.hstack([-lincost_x_vehicle, b_eq_x_vehicle]).T).T
        sol_y_vehicle = jnp.dot(cost_mat_inv_y, jnp.hstack([-lincost_y_vehicle, b_eq_y_vehicle]).T).T

        primal_sol_x_drone = sol_x_drone[:,0:self.nvar]
        primal_sol_y_drone = sol_y_drone[:,0:self.nvar]

        primal_sol_x_vehicle = sol_x_vehicle[:,0:self.nvar]
        primal_sol_y_vehicle = sol_y_vehicle[:,0:self.nvar]

        # Solution
        y_star = jnp.hstack([primal_sol_x_drone, primal_sol_y_drone,
                             primal_sol_x_vehicle, primal_sol_y_vehicle])

        return y_star

    @partial(jit, static_argnums=(0, ))
    # CVAE PointNet - Flax 
    def pointnet(self, pcd, emb_dims=1024, out_features=40, eps=1e-5):

        # PCD Shape Flax - Batch size x Features x Channel, in PyTorch it is B x C x F
        pcd = jnp.transpose(pcd, (0, 2, 1))

        # Conv Layer 1
        variables_1 = {'params': {'kernel': jnp.transpose(self.Wandb_pn[0], (2, 1, 0))}}
        conv1 = nn.Conv(features=64, kernel_size=(1, ), use_bias=False, padding=0)
        out = conv1.apply(variables_1, pcd) 
        out = ((out - self.BN_pn[2]) / jnp.sqrt(self.BN_pn[3] + eps) * self.BN_pn[0]) + self.BN_pn[1]
        out = nn.relu(out)

        # Conv Layer 2
        variables_2 = {'params': {'kernel': jnp.transpose(self.Wandb_pn[1], (2, 1, 0))}}
        conv2 = nn.Conv(features=64, kernel_size=(1, ), use_bias=False, padding=0)
        out = conv2.apply(variables_2, out) 
        out = ((out - self.BN_pn[6]) / jnp.sqrt(self.BN_pn[7] + eps) * self.BN_pn[4]) + self.BN_pn[5]
        out = nn.relu(out)

        # Conv Layer 3
        variables_3 = {'params': {'kernel': jnp.transpose(self.Wandb_pn[2], (2, 1, 0))}}
        conv3 = nn.Conv(features=64, kernel_size=(1, ), use_bias=False, padding=0)
        out = conv3.apply(variables_3, out) 
        out = ((out - self.BN_pn[10]) / jnp.sqrt(self.BN_pn[11] + eps) * self.BN_pn[8]) + self.BN_pn[9]
        out = nn.relu(out)

        # Conv Layer 4
        variables_4 = {'params': {'kernel': jnp.transpose(self.Wandb_pn[3], (2, 1, 0))}}
        conv4 = nn.Conv(features=128, kernel_size=(1, ), use_bias=False, padding=0)
        out = conv4.apply(variables_4, out) 
        out = ((out - self.BN_pn[14]) / jnp.sqrt(self.BN_pn[15] + eps) * self.BN_pn[12]) + self.BN_pn[13]
        out = nn.relu(out)

        # Conv Layer 5
        variables_5 = {'params': {'kernel': jnp.transpose(self.Wandb_pn[4], (2, 1, 0))}}
        conv5 = nn.Conv(features=emb_dims, kernel_size=(1, ), use_bias=False, padding=0) # 1024
        out = conv5.apply(variables_5, out) 
        out = ((out - self.BN_pn[18]) / jnp.sqrt(self.BN_pn[19] + eps) * self.BN_pn[16]) + self.BN_pn[17]
        out = nn.relu(out)

        # Adaptive Maxpool1D
        pool = jax.vmap(equinox.nn.AdaptiveMaxPool1d(1))
        out = pool(jnp.transpose(out, (0, 2, 1))).squeeze()

        # Linear 1
        variables_6 = {'params': {'kernel': jnp.transpose(self.Wandb_pn[5], (1, 0))}}
        fc_1 = nn.Dense(256, use_bias=False)
        out = fc_1.apply(variables_6, out)
        out = ((out - self.BN_pn[22]) / jnp.sqrt(self.BN_pn[23] + eps) * self.BN_pn[20]) + self.BN_pn[21]
        out = nn.relu(out)

        # Linear 2
        variables_7 = {'params': {'kernel': jnp.transpose(self.Wandb_pn[6], (1, 0)), "bias": self.Wandb_pn[7]}}
        fc_2 = nn.Dense(out_features)
        out_fin = fc_2.apply(variables_7, out)

        return out_fin

    @partial(jit, static_argnums=(0, ))
    # CVAE Decoder - Flax
    def decode(self, inp, hidden_dim = 1024 * 2, out_dim=44, eps=1e-5):

        # Layer 1
        variables_wb_1 = {"params": {"kernel": jnp.transpose(self.Wandb_dec[0], (1, 0)), "bias": self.Wandb_dec[1]}}
        fc_1 = nn.Dense(features=hidden_dim)
        out = fc_1.apply(variables_wb_1, inp)
        out = ((out - self.BN_dec[2]) / jnp.sqrt(self.BN_dec[3] + eps) * self.BN_dec[0]) + self.BN_dec[1]
        out = nn.relu(out)

        # Layer 2
        variables_wb_2 = {"params": {"kernel": jnp.transpose(self.Wandb_dec[2], (1, 0)), "bias": self.Wandb_dec[3]}}
        fc_2 = nn.Dense(features=hidden_dim)
        out = fc_2.apply(variables_wb_2, out)
        out = ((out - self.BN_dec[6]) / jnp.sqrt(self.BN_dec[7] + eps) * self.BN_dec[4]) + self.BN_dec[5]
        out = nn.relu(out)

        # Layer 3
        variables_wb_3 = {"params": {"kernel": jnp.transpose(self.Wandb_dec[4], (1, 0)), "bias": self.Wandb_dec[5]}}
        fc_3 = nn.Dense(features=hidden_dim)
        out = fc_3.apply(variables_wb_3, out)
        out = ((out - self.BN_dec[10]) / jnp.sqrt(self.BN_dec[11] + eps) * self.BN_dec[8]) + self.BN_dec[9]
        out = nn.relu(out)

        # Layer 4
        variables_wb_4 = {"params": {"kernel": jnp.transpose(self.Wandb_dec[6], (1, 0)), "bias": self.Wandb_dec[7]}}
        fc_4 = nn.Dense(features=hidden_dim)
        out = fc_4.apply(variables_wb_4, out)
        out = ((out - self.BN_dec[14]) / jnp.sqrt(self.BN_dec[15] + eps) * self.BN_dec[12]) + self.BN_dec[13]
        out = nn.relu(out)

        # Layer 5
        variables_wb_5 = {"params": {"kernel": jnp.transpose(self.Wandb_dec[8], (1, 0)), "bias": self.Wandb_dec[9]}}
        fc_5 = nn.Dense(features=256)
        out = fc_5.apply(variables_wb_5, out)
        out = ((out - self.BN_dec[18]) / jnp.sqrt(self.BN_dec[19] + eps) * self.BN_dec[16]) + self.BN_dec[17]
        out = nn.relu(out)

        # Layer 6 Final
        variables_wb_6 = {"params": {"kernel": jnp.transpose(self.Wandb_dec[10], (1, 0)), "bias": self.Wandb_dec[11]}}
        fc_6 = nn.Dense(features=out_dim)
        out_fin = fc_6.apply(variables_wb_6, out)
        
        return out_fin  

    @partial(jit, static_argnums=(0, ))
    def forward(self, state, pcd):

        # Normalize states & Scale Point cloud
        state_norm = (state - self.state_mean) / self.state_std
        scaled_pcd = (pcd - self.min_pcd) / (self.max_pcd - self.min_pcd)

        # Inference with cVAE
        batch_init_state = jnp.vstack([state[:,0:10]] * self.num_batch)
        batch_state_norm = jnp.vstack([state_norm] * self.num_batch)
        batch_pcd_scaled = jnp.vstack([scaled_pcd] * self.num_batch)
        
        z = jax.random.normal(self.key, (self.num_batch, self.zdim)) 

        # PCD feature extractor
        pcd_features = self.pointnet(batch_pcd_scaled)

        inputs = jnp.hstack([z, batch_state_norm, pcd_features])
        y = self.decode(inputs)

        # Call Optimization Solver 
        y_star = self.solve(batch_init_state, y)

        return y_star