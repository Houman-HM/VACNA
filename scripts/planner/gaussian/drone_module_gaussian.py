#!/usr/bin/python
import numpy as np
import jax.numpy as jnp
from functools import partial
from jax import jit, random
import bernstein_coeff_order10_arbitinterval
import time
import matplotlib.pyplot as plt 
import jax

def get_weights_biases(weight_biases_mat_file):
    W0, b0, W1, b1, W2, b2, W3, b3 =  weight_biases_mat_file['w0'], weight_biases_mat_file['b0'], \
                                        weight_biases_mat_file['w1'], weight_biases_mat_file['b1'], \
                                    weight_biases_mat_file['w2'], weight_biases_mat_file['b2'], \
                                    weight_biases_mat_file['w3'], weight_biases_mat_file['b3']
    
    return jnp.asarray(W0), jnp.asarray(b0), jnp.asarray(W1), jnp.asarray(b1), \
            jnp.asarray(W2), jnp.asarray(b2), jnp.asarray(W3), jnp.asarray(b3)

class batch_occ_tracking():

	def __init__(self, P, Pdot, Pddot, v_max, a_max, t_fin, num, num_batch_projection,
					 tot_time, rho_ineq, maxiter_projection, rho_projection, 
					 rho_tracking, rho_obs, maxiter_cem, d_min_tracking, d_max_tracking,
					P_up_jax, Pdot_up_jax, Pddot_up_jax, occlusion_weight, ellite_num_projection, ellite_num,
					initial_up_sampling, target_distance_weight, smoothness_weight, a_obs, b_obs, num_obs,
					d_min_tracking_vehicle, d_max_tracking_vehicle):
		
		self.rho_ineq = rho_ineq
		self.rho_projection = rho_projection
		self.rho_tracking = rho_tracking
		self.rho_obs = rho_obs
		self.maxiter_projection = maxiter_projection
		self.maxiter_cem = maxiter_cem
		self.occlusion_weight = occlusion_weight
		self.num_tracking = 1
		self.initial_up_sampling = initial_up_sampling

		self.target_distance_weight = target_distance_weight
		self.smoothness_weight = smoothness_weight
		
		self.a_obs = a_obs
		self.b_obs = b_obs
		self.num_obs = num_obs

		self.d_min_tracking = d_min_tracking
		self.d_max_tracking = d_max_tracking

		self.d_min_tracking_vehicle = d_min_tracking_vehicle
		self.d_max_tracking_vehicle = d_max_tracking_vehicle

		self.t_fin = t_fin
		self.num = num
		self.t = self.t_fin/self.num
		self.num_batch_projection = num_batch_projection

		self.v_max = v_max
		self.a_max = a_max
	
		self.tot_time = tot_time

		self.P = P
		self.Pdot = Pdot
		self.Pddot = Pddot
		self.ellite_num_projection = ellite_num_projection
		self.ellite_num = ellite_num

		self.P_jax, self.Pdot_jax, self.Pddot_jax = jnp.asarray(self.P), jnp.asarray(self.Pdot), jnp.asarray(self.Pddot)
		self.nvar = jnp.shape(self.P_jax)[1]
	
		self.A_eq = jnp.vstack((self.P_jax[0], self.Pdot_jax[0], self.Pddot_jax[0], self.Pdot_jax[-1], self.Pddot_jax[-1] ))
			
		self.A_vel = self.Pdot_jax 
		self.A_acc = self.Pddot_jax
		self.A_projection = jnp.identity(self.nvar)		

		self.A_tracking = self.P_jax
		self.A_tracking_vehicle = self.P_jax
		self.A_workspace = self.P_jax
		self.P_up_jax = P_up_jax
		self.Pdot_up_jax = Pdot_up_jax
		self.Pddot_up_jax = Pddot_up_jax

		self.W0 = None
		self.b0 = None
		self.W1 = None 
		self.b1 = None
		self.W2 = None
		self.b2 = None
		self.W3 = None
		self.b3 = None
		self.x_obs = None
		self.y_obs = None
		self.obstacle_points = None
		self.A_obs = jnp.tile(self.P_jax, (self.num_obs, 1))


		A = np.diff(np.diff(np.identity(self.num), axis = 0), axis = 0)

		temp_1 = np.zeros(self.num)
		temp_2 = np.zeros(self.num)
		temp_3 = np.zeros(self.num)
		temp_4 = np.zeros(self.num)

		temp_1[0] = 1.0
		temp_2[0] = -2
		temp_2[1] = 1
		temp_3[-1] = -2
		temp_3[-2] = 1

		temp_4[-1] = 1.0

		A_mat = -np.vstack(( temp_1, temp_2, A, temp_3, temp_4   ))
		# A_mat = A

		R = np.dot(A_mat.T, A_mat)
		mu = np.zeros(self.num)
		cov = np.linalg.pinv(R)
		self.mu = jnp.asarray(mu)
		################# Gaussian Trajectory Sampling
		self.cov = jnp.asarray(cov)


		eps_k = np.random.multivariate_normal(mu, 0.1*cov, (int(self.num_batch_projection/1), ))
		eps_k_up_sampling = np.random.multivariate_normal(mu, 0.001*cov, (int((self.num_batch_projection * self.initial_up_sampling)), ))
		self.eps_k = jnp.asarray(eps_k)
		self.eps_k_up_sampling = jnp.asarray(eps_k_up_sampling)


		self.cost = self.rho_projection * jnp.dot(self.A_projection.T, self.A_projection) + self.rho_ineq * jnp.dot(self.A_vel.T, self.A_vel)+ \
					self.rho_ineq * jnp.dot(self.A_acc.T, self.A_acc) + self.rho_tracking * jnp.dot(self.A_tracking.T, self.A_tracking) +\
					self.rho_obs * jnp.dot(self.A_obs.T, self.A_obs) + self.rho_tracking * jnp.dot(self.A_tracking_vehicle.T, self.A_tracking_vehicle)

		self.cost_matrix = jnp.vstack(( jnp.hstack(( self.cost, self.A_eq.T )), jnp.hstack((self.A_eq, jnp.zeros(( jnp.shape(self.A_eq)[0], jnp.shape(self.A_eq)[0] )) )) ))

		self.discouting_factor = (jnp.linspace(0.1,1, self.num) * 10).reshape(1, self.num)
		self.discouting_factor =1

	@partial(jit, static_argnums=(0,))
	def initial_alpha_d_obs(self, x_samples_init, y_samples_init,
								xdot_samples_init, ydot_samples_init,\
                        		xddot_samples_init, yddot_samples_init,\
								x_tracking, y_tracking, x_obs_traj, y_obs_traj,
								x_tracking_vehicle, y_tracking_vehicle):

		wc_alpha_tracking = (x_samples_init-x_tracking[:,jnp.newaxis])
		ws_alpha_tracking = (y_samples_init-y_tracking[:,jnp.newaxis])

		wc_alpha_tracking = wc_alpha_tracking.transpose(1, 0, 2)
		ws_alpha_tracking = ws_alpha_tracking.transpose(1, 0, 2)

		wc_alpha_tracking = wc_alpha_tracking.reshape(self.num_batch_projection, self.num*self.num_tracking)
		ws_alpha_tracking = ws_alpha_tracking.reshape(self.num_batch_projection, self.num*self.num_tracking)

		alpha_tracking = jnp.arctan2( ws_alpha_tracking, wc_alpha_tracking)
		c1_d = 1.0*self.rho_tracking*(jnp.cos(alpha_tracking)**2 + jnp.sin(alpha_tracking)**2 )
		c2_d = 1.0*self.rho_tracking*(wc_alpha_tracking*jnp.cos(alpha_tracking) + ws_alpha_tracking*jnp.sin(alpha_tracking)  )

		d_tracking = c2_d/c1_d	
		d_tracking = jnp.clip( d_tracking, self.d_min_tracking*jnp.ones((self.num_batch_projection,  self.num*self.num_tracking   )), self.d_max_tracking*jnp.ones((self.num_batch_projection,  self.num*self.num_tracking   ))  )

		
		wc_alpha_tracking_vehicle = (x_samples_init-x_tracking_vehicle[:,jnp.newaxis])
		ws_alpha_tracking_vehicle = (y_samples_init-y_tracking_vehicle[:,jnp.newaxis])

		wc_alpha_tracking_vehicle = wc_alpha_tracking_vehicle.transpose(1, 0, 2)
		ws_alpha_tracking_vehicle = ws_alpha_tracking_vehicle.transpose(1, 0, 2)

		wc_alpha_tracking_vehicle = wc_alpha_tracking_vehicle.reshape(self.num_batch_projection, self.num*self.num_tracking)
		ws_alpha_tracking_vehicle = ws_alpha_tracking_vehicle.reshape(self.num_batch_projection, self.num*self.num_tracking)

		alpha_tracking_vehicle = jnp.arctan2( ws_alpha_tracking_vehicle, wc_alpha_tracking_vehicle)
		c1_d_vehicle = 1.0*self.rho_tracking*(jnp.cos(alpha_tracking_vehicle)**2 + jnp.sin(alpha_tracking_vehicle)**2 )
		c2_d_vehicle = 1.0*self.rho_tracking*(wc_alpha_tracking_vehicle*jnp.cos(alpha_tracking_vehicle) + ws_alpha_tracking_vehicle*jnp.sin(alpha_tracking_vehicle)  )

		d_tracking_vehicle = c2_d_vehicle/c1_d_vehicle	
		d_tracking_vehicle = jnp.clip( d_tracking_vehicle, self.d_min_tracking_vehicle*jnp.ones((self.num_batch_projection,  self.num*self.num_tracking   )), self.d_max_tracking_vehicle*jnp.ones((self.num_batch_projection,  self.num*self.num_tracking   ))  )

		
		
		
		
		wc_alpha_obs = (x_samples_init - x_obs_traj[:,jnp.newaxis]).transpose(1, 0, 2)
		ws_alpha_obs = (y_samples_init - y_obs_traj[:,jnp.newaxis]).transpose(1, 0, 2)
		wc_alpha_obs = wc_alpha_obs.reshape(self.num_batch_projection, self.num * self.num_obs)
		ws_alpha_obs = ws_alpha_obs.reshape(self.num_batch_projection, self.num * self.num_obs)
		alpha_obs = jnp.arctan2(ws_alpha_obs * self.a_obs, wc_alpha_obs * self.b_obs)

		c1_d = 1.0*self.rho_obs*(self.a_obs**2*jnp.cos(alpha_obs)**2 + self.b_obs**2*jnp.sin(alpha_obs)**2 )
		c2_d = 1.0*self.rho_obs*(self.a_obs*wc_alpha_obs*jnp.cos(alpha_obs) + self.b_obs * ws_alpha_obs * jnp.sin(alpha_obs)  )

		d_obs = c2_d/c1_d
		d_obs = jnp.maximum(jnp.ones(d_obs.shape), d_obs)


		wc_alpha_vx = xdot_samples_init
		ws_alpha_vy = ydot_samples_init
		alpha_v = jnp.arctan2( ws_alpha_vy, wc_alpha_vx)		

		c1_d_v = 1.0*self.rho_ineq*(jnp.cos(alpha_v)**2 + jnp.sin(alpha_v)**2 )
		c2_d_v = 1.0*self.rho_ineq*(wc_alpha_vx*jnp.cos(alpha_v) + ws_alpha_vy*jnp.sin(alpha_v)  )

		d_temp_v = c2_d_v/c1_d_v

		d_v = jnp.minimum(self.v_max*jnp.ones((self.num_batch_projection, self.num)), d_temp_v   )
		
		################# acceleration terms

		wc_alpha_ax = xddot_samples_init
		ws_alpha_ay = yddot_samples_init
		alpha_a = jnp.arctan2( ws_alpha_ay, wc_alpha_ax)		
		c1_d_a = 1.0*self.rho_ineq * (jnp.cos(alpha_a)**2 + jnp.sin(alpha_a)**2 )
		c2_d_a = 1.0*self.rho_ineq * (wc_alpha_ax * jnp.cos(alpha_a) + ws_alpha_ay * jnp.sin(alpha_a)  )
		d_temp_a = c2_d_a/c1_d_a
		d_a = jnp.minimum(self.a_max*jnp.ones((self.num_batch_projection, self.num)), d_temp_a   )

		return alpha_tracking, d_tracking, alpha_obs, d_obs, alpha_v, d_v, alpha_a, d_a, alpha_tracking_vehicle, d_tracking_vehicle


	@partial(jit, static_argnums = (0,))
	def compute_projection(self, x_samples_init, y_samples_init, b_x_eq, b_y_eq, x_tracking, y_tracking,
						 alpha_tracking, d_tracking, alpha_a, d_a, alpha_v, d_v,
						lamda_x, lamda_y, c_x_samples_init, c_y_samples_init, alpha_obs, d_obs, x_obs_traj, y_obs_traj,
						alpha_tracking_vehicle, d_tracking_vehicle, x_tracking_vehicle, y_tracking_vehicle):
		
		temp_x_tracking = d_tracking*jnp.cos(alpha_tracking)
		b_tracking_x = x_tracking.reshape(self.num*self.num_tracking)+temp_x_tracking
		 
		temp_y_tracking = d_tracking*jnp.sin(alpha_tracking)
		b_tracking_y = y_tracking.reshape(self.num*self.num_tracking)+temp_y_tracking


		temp_x_tracking_vehicle = d_tracking_vehicle*jnp.cos(alpha_tracking_vehicle)
		b_tracking_x_vehicle = x_tracking_vehicle.reshape(self.num*self.num_tracking)+temp_x_tracking_vehicle
		 
		temp_y_tracking_vehicle = d_tracking_vehicle*jnp.sin(alpha_tracking_vehicle)
		b_tracking_y_vehicle = y_tracking_vehicle.reshape(self.num*self.num_tracking)+temp_y_tracking_vehicle

		temp_x_obs = d_obs * jnp.cos(alpha_obs) * self.a_obs
		b_obs_x = x_obs_traj.reshape(self.num*self.num_obs)+temp_x_obs
			
		temp_y_obs = d_obs*jnp.sin(alpha_obs)*self.b_obs
		b_obs_y = y_obs_traj.reshape(self.num*self.num_obs)+temp_y_obs

		b_ax_ineq = d_a*jnp.cos(alpha_a)
		b_ay_ineq = d_a*jnp.sin(alpha_a)

		b_vx_ineq = d_v*jnp.cos(alpha_v)
		b_vy_ineq = d_v*jnp.sin(alpha_v)

		b_x_projection = c_x_samples_init
		b_y_projection = c_y_samples_init
		
		lincost_x = -lamda_x-self.rho_projection*jnp.dot(self.A_projection.T, b_x_projection.T).T-self.rho_ineq*jnp.dot(self.A_acc.T, b_ax_ineq.T).T\
				-self.rho_ineq * jnp.dot(self.A_vel.T, b_vx_ineq.T).T - self.rho_tracking * jnp.dot(self.A_tracking.T, b_tracking_x.T).T\
				-self.rho_obs * jnp.dot(self.A_obs.T, b_obs_x.T).T  - self.rho_tracking * jnp.dot(self.A_tracking_vehicle.T, b_tracking_x_vehicle.T).T
		
		lincost_y = -lamda_y-self.rho_projection*jnp.dot(self.A_projection.T, b_y_projection.T).T-self.rho_ineq*jnp.dot(self.A_acc.T, b_ay_ineq.T).T\
					-self.rho_ineq*jnp.dot(self.A_vel.T, b_vy_ineq.T).T - self.rho_tracking*jnp.dot(self.A_tracking.T, b_tracking_y.T).T\
					-self.rho_obs * jnp.dot(self.A_obs.T, b_obs_y.T).T - self.rho_tracking*jnp.dot(self.A_tracking_vehicle.T, b_tracking_y_vehicle.T).T

		sol_x_temp = jnp.linalg.solve(self.cost_matrix, jnp.hstack((-lincost_x, b_x_eq )).T ).T
		sol_y_temp = jnp.linalg.solve(self.cost_matrix, jnp.hstack((-lincost_y, b_y_eq )).T ).T

		c_x_samples = sol_x_temp[:, 0:self.nvar]
		c_y_samples = sol_y_temp[:, 0:self.nvar]

		x_samples = jnp.dot(self.P_jax, c_x_samples.T).T 
		y_samples = jnp.dot(self.P_jax, c_y_samples.T).T

		xdot_samples = jnp.dot(self.Pdot_jax, c_x_samples.T).T 
		ydot_samples = jnp.dot(self.Pdot_jax, c_y_samples.T).T

		xddot_samples = jnp.dot(self.Pddot_jax, c_x_samples.T).T 
		yddot_samples = jnp.dot(self.Pddot_jax, c_y_samples.T).T

		return c_x_samples, c_y_samples, x_samples, y_samples, xdot_samples, ydot_samples, xddot_samples, yddot_samples


	@partial(jit, static_argnums=(0,))	
	def compute_alph_d(self, x_samples, y_samples, xdot_samples,
					 ydot_samples, xddot_samples, yddot_samples,
					  x_tracking, y_tracking, lamda_x, lamda_y, x_obs_traj, y_obs_traj,
					  x_tracking_vehicle, y_tracking_vehicle):

		#################################### tracking

		wc_alpha_tracking = (x_samples-x_tracking[:,jnp.newaxis])
		ws_alpha_tracking = (y_samples-y_tracking[:,jnp.newaxis])

		wc_alpha_tracking = wc_alpha_tracking.transpose(1, 0, 2)
		ws_alpha_tracking = ws_alpha_tracking.transpose(1, 0, 2)

		wc_alpha_tracking = wc_alpha_tracking.reshape(self.num_batch_projection, self.num)
		ws_alpha_tracking = ws_alpha_tracking.reshape(self.num_batch_projection, self.num)

		alpha_tracking = jnp.arctan2( ws_alpha_tracking, wc_alpha_tracking)
		c1_d = 1.0*self.rho_tracking*(jnp.cos(alpha_tracking)**2 + jnp.sin(alpha_tracking)**2 )
		c2_d = 1.0*self.rho_tracking*(wc_alpha_tracking*jnp.cos(alpha_tracking) + ws_alpha_tracking*jnp.sin(alpha_tracking)  )

		d_tracking = c2_d/c1_d
		d_tracking = jnp.clip( d_tracking, self.d_min_tracking*jnp.ones(d_tracking.shape), self.d_max_tracking * jnp.ones(d_tracking.shape)  )

		####################### Vehicle Tracking
		
		wc_alpha_tracking_vehicle = (x_samples-x_tracking_vehicle[:,jnp.newaxis])
		ws_alpha_tracking_vehicle = (y_samples-y_tracking_vehicle[:,jnp.newaxis])

		wc_alpha_tracking_vehicle = wc_alpha_tracking_vehicle.transpose(1, 0, 2)
		ws_alpha_tracking_vehicle = ws_alpha_tracking_vehicle.transpose(1, 0, 2)

		wc_alpha_tracking_vehicle = wc_alpha_tracking_vehicle.reshape(self.num_batch_projection, self.num)
		ws_alpha_tracking_vehicle = ws_alpha_tracking_vehicle.reshape(self.num_batch_projection, self.num)

		alpha_tracking_vehicle = jnp.arctan2(ws_alpha_tracking_vehicle, wc_alpha_tracking_vehicle)
		c1_d_vehicle = 1.0*self.rho_tracking*(jnp.cos(alpha_tracking_vehicle)**2 + jnp.sin(alpha_tracking_vehicle)**2 )
		c2_d_vehicle = 1.0*self.rho_tracking*(wc_alpha_tracking_vehicle*jnp.cos(alpha_tracking_vehicle) + ws_alpha_tracking_vehicle*jnp.sin(alpha_tracking_vehicle)  )

		d_tracking_vehicle = c2_d_vehicle/c1_d_vehicle
		d_tracking_vehicle = jnp.clip( d_tracking_vehicle, self.d_min_tracking_vehicle*jnp.ones(d_tracking_vehicle.shape), self.d_max_tracking_vehicle * jnp.ones(d_tracking_vehicle.shape)  )


		wc_alpha_vx = xdot_samples
		ws_alpha_vy = ydot_samples
		alpha_v = jnp.arctan2( ws_alpha_vy, wc_alpha_vx)		

		c1_d_v = 1.0*self.rho_ineq*(jnp.cos(alpha_v)**2 + jnp.sin(alpha_v)**2 )
		c2_d_v = 1.0*self.rho_ineq*(wc_alpha_vx*jnp.cos(alpha_v) + ws_alpha_vy*jnp.sin(alpha_v)  )

		d_temp_v = c2_d_v/c1_d_v

		d_v = jnp.minimum(self.v_max*jnp.ones((self.num_batch_projection, self.num)), d_temp_v   )
		
		################# acceleration terms

		wc_alpha_ax = xddot_samples
		ws_alpha_ay = yddot_samples
		alpha_a = jnp.arctan2( ws_alpha_ay, wc_alpha_ax)		
		c1_d_a = 1.0*self.rho_ineq * (jnp.cos(alpha_a)**2 + jnp.sin(alpha_a)**2 )
		c2_d_a = 1.0*self.rho_ineq * (wc_alpha_ax * jnp.cos(alpha_a) + ws_alpha_ay * jnp.sin(alpha_a)  )
		d_temp_a = c2_d_a/c1_d_a
		d_a = jnp.minimum(self.a_max*jnp.ones((self.num_batch_projection, self.num)), d_temp_a   )

		wc_alpha_obs = (x_samples - x_obs_traj[:,jnp.newaxis]).transpose(1, 0, 2)
		ws_alpha_obs = (y_samples - y_obs_traj[:,jnp.newaxis]).transpose(1, 0, 2)
		wc_alpha_obs = wc_alpha_obs.reshape(self.num_batch_projection, self.num * self.num_obs)
		ws_alpha_obs = ws_alpha_obs.reshape(self.num_batch_projection, self.num * self.num_obs)
		alpha_obs = jnp.arctan2(ws_alpha_obs * self.a_obs, wc_alpha_obs * self.b_obs)

		c1_d = 1.0*self.rho_obs*(self.a_obs**2*jnp.cos(alpha_obs)**2 + self.b_obs**2*jnp.sin(alpha_obs)**2 )
		c2_d = 1.0*self.rho_obs*(self.a_obs*wc_alpha_obs*jnp.cos(alpha_obs) + self.b_obs * ws_alpha_obs * jnp.sin(alpha_obs))

		d_obs = c2_d/c1_d
		d_obs = jnp.maximum(jnp.ones(d_obs.shape), d_obs)


		#########################################33
		res_ax_vec = xddot_samples-d_a*jnp.cos(alpha_a)
		res_ay_vec = yddot_samples-d_a*jnp.sin(alpha_a)

		res_vx_vec = xdot_samples-d_v*jnp.cos(alpha_v)
		res_vy_vec = ydot_samples-d_v*jnp.sin(alpha_v)

		res_x_tracking_vec = wc_alpha_tracking-d_tracking*jnp.cos(alpha_tracking)
		res_y_tracking_vec = ws_alpha_tracking-d_tracking*jnp.sin(alpha_tracking)

		res_x_tracking_vehicle_vec = wc_alpha_tracking_vehicle-d_tracking_vehicle*jnp.cos(alpha_tracking_vehicle)
		res_y_tracking_vehicle_vec = ws_alpha_tracking_vehicle-d_tracking_vehicle*jnp.sin(alpha_tracking_vehicle)

		res_x_obs_vec = wc_alpha_obs - self.a_obs * d_obs * jnp.cos(alpha_obs)
		res_y_obs_vec = ws_alpha_obs - self.b_obs * d_obs * jnp.sin(alpha_obs)

		lamda_x = lamda_x-self.rho_ineq*jnp.dot(self.A_acc.T, res_ax_vec.T).T-self.rho_ineq*jnp.dot(self.A_vel.T, res_vx_vec.T).T\
				-self.rho_tracking*jnp.dot(self.A_tracking.T, res_x_tracking_vec.T).T - self.rho_obs * jnp.dot(self.A_obs.T, res_x_obs_vec.T).T\
				-self.rho_tracking*jnp.dot(self.A_tracking_vehicle.T, res_x_tracking_vehicle_vec.T).T

		lamda_y = lamda_y-self.rho_ineq*jnp.dot(self.A_acc.T, res_ay_vec.T).T-self.rho_ineq*jnp.dot(self.A_vel.T, res_vy_vec.T).T\
				-self.rho_tracking*jnp.dot(self.A_tracking.T, res_y_tracking_vec.T).T - self.rho_obs * jnp.dot(self.A_obs.T, res_y_obs_vec.T).T\
				-self.rho_tracking*jnp.dot(self.A_tracking_vehicle.T, res_y_tracking_vehicle_vec.T).T

		res_tracking_vec = jnp.hstack(( res_x_tracking_vec, res_y_tracking_vec  ))
		res_tracking_vehicle_vec = jnp.hstack(( res_x_tracking_vehicle_vec, res_y_tracking_vehicle_vec  ))

		res_obs_vec = jnp.hstack(( res_x_obs_vec, res_y_obs_vec))
		
		res_acc_vec = jnp.hstack(( res_ax_vec,  res_ay_vec  ))
		res_vel_vec = jnp.hstack(( res_vx_vec,  res_vy_vec  ))

		res_norm_batch = jnp.linalg.norm(res_acc_vec, axis =1) + jnp.linalg.norm(res_vel_vec, axis =1) + jnp.linalg.norm(res_tracking_vec, axis =1)\
			+ jnp.linalg.norm(res_obs_vec, axis =1) + jnp.linalg.norm(res_tracking_vehicle_vec, axis =1)

		res_tracking_vec = jnp.linalg.norm(res_tracking_vec, axis=1)
		res_acc_norm = jnp.linalg.norm(res_acc_vec, axis =1)
		res_vel_norm = jnp.linalg.norm(res_vel_vec, axis =1)
		res_obs_norm = jnp.linalg.norm(res_obs_vec, axis =1)
	
		return alpha_tracking, d_tracking, alpha_a, d_a, alpha_v, d_v, lamda_x, lamda_y,\
			 res_tracking_vec, res_acc_vec, res_vel_vec, res_tracking_vec, res_acc_norm, res_vel_norm, res_norm_batch, \
			alpha_obs, d_obs, res_obs_norm, d_tracking_vehicle, alpha_tracking_vehicle
	

	@partial(jit, static_argnums=(0,))	
	def compute_boundary_vec(self, x_init, vx_init, ax_init, y_init, vy_init, ay_init, vx_tracking, vy_tracking):

		x_init_vec = x_init*jnp.ones((self.num_batch_projection, 1))
		y_init_vec = y_init*jnp.ones((self.num_batch_projection, 1)) 

		vx_init_vec = vx_init*jnp.ones((self.num_batch_projection, 1))
		vy_init_vec = vy_init*jnp.ones((self.num_batch_projection, 1))

		ax_init_vec = ax_init*jnp.ones((self.num_batch_projection, 1))
		ay_init_vec = ay_init*jnp.ones((self.num_batch_projection, 1))

		vx_fin_vec = 0*jnp.ones((self.num_batch_projection, 1))
		vy_fin_vec = 0*jnp.ones((self.num_batch_projection, 1))

		ax_fin_vec = 0.0*jnp.ones((self.num_batch_projection, 1))
		ay_fin_vec = 0*jnp.ones((self.num_batch_projection, 1))
		
		b_eq_x = jnp.hstack(( x_init_vec, vx_init_vec, ax_init_vec, vx_fin_vec, ax_fin_vec ))
		b_eq_y = jnp.hstack(( y_init_vec, vy_init_vec, ay_init_vec, vy_fin_vec, ay_fin_vec ))
		
		return b_eq_x, b_eq_y
	

	@partial(jit, static_argnums=(0,))	
	def compute_initial_samples(self, x_init_vehicle, y_init_vehicle,
									x_fin_drone, y_fin_drone, obstacle_pointcloud):

		x_fin_vehicle = x_fin_drone
		y_fin_vehicle = y_fin_drone

		goal_rot_vehicle = -jnp.arctan2(y_fin_vehicle - y_init_vehicle, x_fin_vehicle - x_init_vehicle)
		x_init_temp_vehicle = x_init_vehicle * jnp.cos(goal_rot_vehicle) - y_init_vehicle * jnp.sin(goal_rot_vehicle)
		y_init_temp_vehicle = x_init_vehicle * jnp.sin(goal_rot_vehicle) + y_init_vehicle * jnp.cos(goal_rot_vehicle)
		x_fin_temp_vehicle = x_fin_vehicle * jnp.cos(goal_rot_vehicle) - y_fin_vehicle * jnp.sin(goal_rot_vehicle)
		y_fin_temp_vehicle = x_fin_vehicle * jnp.sin(goal_rot_vehicle) + y_fin_vehicle * jnp.cos(goal_rot_vehicle)

		x_interp_vehicle = jnp.linspace(x_init_temp_vehicle, x_fin_temp_vehicle, self.num)
		y_interp_vehicle = jnp.linspace(y_init_temp_vehicle, y_fin_temp_vehicle, self.num)

		x_guess_temp_vehicle = x_interp_vehicle + 0.0 * self.eps_k_up_sampling 
		y_guess_temp_vehicle = y_interp_vehicle + self.eps_k_up_sampling

		x_guess_sampling_vehicle = x_guess_temp_vehicle * jnp.cos(goal_rot_vehicle) + y_guess_temp_vehicle * jnp.sin(goal_rot_vehicle)
		y_guess_sampling_vehicle = - x_guess_temp_vehicle * jnp.sin(goal_rot_vehicle) + y_guess_temp_vehicle * jnp.cos(goal_rot_vehicle)

		x_obs = jnp.repeat(obstacle_pointcloud[:,0], self.num).reshape(obstacle_pointcloud.shape[0],self.num)
		y_obs = jnp.repeat(obstacle_pointcloud[:,1], self.num).reshape(obstacle_pointcloud.shape[0],self.num)

		wc_alpha_temp_vehicle = (x_guess_sampling_vehicle-x_obs[:,jnp.newaxis])
		ws_alpha_temp_vehicle = (y_guess_sampling_vehicle-y_obs[:,jnp.newaxis])
		wc_alpha_vehicle = wc_alpha_temp_vehicle.transpose(1, 0, 2)
		ws_alpha_vehicle = ws_alpha_temp_vehicle.transpose(1, 0, 2)
		wc_alpha_vehicle = wc_alpha_vehicle.reshape(self.num_batch_projection * self.initial_up_sampling, self.num * x_obs.shape[0])
		ws_alpha_vehicle = ws_alpha_vehicle.reshape(self.num_batch_projection * self.initial_up_sampling, self.num * y_obs.shape[0])

		dist_obs_vehicle = -wc_alpha_vehicle**2/(self.a_obs**2) -ws_alpha_vehicle**2/(self.b_obs**2)+ 1
		dist_obs = dist_obs_vehicle

		cost_obs_penalty = jnp.linalg.norm(jnp.maximum(jnp.zeros((self.initial_up_sampling * self.num_batch_projection, self.num * x_obs.shape[0])), dist_obs), axis = 1)
		idx_ellite = jnp.argsort(cost_obs_penalty)

		x_guess_vehicle_temp = x_guess_sampling_vehicle[idx_ellite[0:self.num_batch_projection]]
		y_guess_vehicle_temp = y_guess_sampling_vehicle[idx_ellite[0:self.num_batch_projection]]

		########
		cost_regression = jnp.dot(self.P_jax.T, self.P_jax) + 0.0001 * jnp.identity(self.nvar)
		lincost_regression_x_vehicle = -jnp.dot(self.P_jax.T, x_guess_vehicle_temp.T).T 
		lincost_regression_y_vehicle = -jnp.dot(self.P_jax.T, y_guess_vehicle_temp.T).T

		cost_mat_inv = jnp.linalg.inv(cost_regression)

		c_x_samples_init_vehicle = jnp.dot(cost_mat_inv, -lincost_regression_x_vehicle.T).T 
		c_y_samples_init_vehicle = jnp.dot(cost_mat_inv, -lincost_regression_y_vehicle.T).T

		x_guess_vehicle = jnp.dot(self.P_jax, c_x_samples_init_vehicle.T).T
		y_guess_vehicle = jnp.dot(self.P_jax, c_y_samples_init_vehicle.T).T

		xdot_samples_init = jnp.dot(self.Pdot_jax, c_x_samples_init_vehicle.T).T
		ydot_samples_init = jnp.dot(self.Pdot_jax, c_y_samples_init_vehicle.T).T

		xddot_samples_init = jnp.dot(self.Pddot_jax, c_x_samples_init_vehicle.T).T
		yddot_samples_init = jnp.dot(self.Pddot_jax, c_y_samples_init_vehicle.T).T

		return c_x_samples_init_vehicle, c_y_samples_init_vehicle, x_guess_vehicle, y_guess_vehicle,\
				 xdot_samples_init, ydot_samples_init, xddot_samples_init, yddot_samples_init

	@partial(jit, static_argnums = (0,) )
	def compute_projection_samples(self, x_init, vx_init, ax_init, y_init, vy_init, ay_init, \
								x_tracking, y_tracking, lamda_x, lamda_y, x_samples_init, y_samples_init, 
									 c_x_samples_init, c_y_samples_init, xdot_samples_init, ydot_samples_init,\
                        			xddot_samples_init, yddot_samples_init,
									vx_tracking, vy_tracking, x_obs_traj, y_obs_traj, x_tracking_vehicle, y_tracking_vehicle):

		b_x_eq, b_y_eq = self.compute_boundary_vec(x_init, vx_init, ax_init, y_init, vy_init, ay_init, vx_tracking, vy_tracking)

		alpha_tracking, d_tracking, alpha_obs, d_obs, alpha_v, d_v, alpha_a, d_a, alpha_tracking_vehicle, d_tracking_vehicle = self.initial_alpha_d_obs(x_samples_init, y_samples_init, 
																				xdot_samples_init, ydot_samples_init,\
                        														xddot_samples_init, yddot_samples_init, \
																				x_tracking, y_tracking, x_obs_traj, y_obs_traj,
																				x_tracking_vehicle, y_tracking_vehicle)
		
		for i in range(0, self.maxiter_projection):

			c_x_samples, c_y_samples, x_samples, y_samples, xdot_samples, \
				ydot_samples, xddot_samples, yddot_samples = self.compute_projection(x_samples_init, y_samples_init, b_x_eq, b_y_eq, x_tracking, 
				y_tracking, alpha_tracking, d_tracking, alpha_a, d_a, alpha_v, d_v, lamda_x, 
				lamda_y, c_x_samples_init, c_y_samples_init, alpha_obs, d_obs, x_obs_traj, y_obs_traj,
				alpha_tracking_vehicle, d_tracking_vehicle, x_tracking_vehicle, y_tracking_vehicle)
		
			alpha_tracking, d_tracking, alpha_a, d_a, alpha_v, d_v, lamda_x, lamda_y, \
				res_tracking_vec, res_acc_vec, res_vel_vec, res_tracking_vec, \
					 res_acc_norm, res_vel_norm, res_norm_batch, alpha_obs, d_obs, res_obs_norm, d_tracking_vehicle, alpha_tracking_vehicle = self.compute_alph_d(x_samples, y_samples, xdot_samples, ydot_samples,
					  xddot_samples, yddot_samples, x_tracking, y_tracking, lamda_x, lamda_y, x_obs_traj, y_obs_traj,
					  x_tracking_vehicle, y_tracking_vehicle)
		
		return x_samples, y_samples, xdot_samples, ydot_samples, xddot_samples, \
			yddot_samples, c_x_samples, c_y_samples, alpha_v, d_v, alpha_a, d_a, alpha_tracking, \
				d_tracking, lamda_x, lamda_y, res_norm_batch, res_obs_norm
 	   
	@partial(jit, static_argnums=(0,))	

	def compute_cost_batch(self, xddot_samples, yddot_samples, x_samples, y_samples,
							 d_avg_tracking, x_rack_traj, y_rack_traj, obstacle_points, res_norm_projection,
							 x_best_vehicle, y_best_vehicle): 

		mu =  2.3333
		std = 6.0117

		tiled_obstacle_points = jnp.tile(obstacle_points, (self.num * self.ellite_num_projection,1))
		tiled_tracking_trajectory_x = jnp.tile(x_best_vehicle, (self.ellite_num_projection))
		tiled_tracking_trajectory_y = jnp.tile(y_best_vehicle, (self.ellite_num_projection))

		tracking_robot_matrix  = jnp.hstack((x_samples.reshape(self.ellite_num_projection * self.num, 1), y_samples.reshape(self.ellite_num_projection * self.num, 1), 
											tiled_tracking_trajectory_x.reshape(self.ellite_num_projection * self.num, 1),
												tiled_tracking_trajectory_y.reshape(self.ellite_num_projection * self.num, 1)))
		tiled_tracking_robot_matrix = jnp.repeat(tracking_robot_matrix, (obstacle_points.shape)[0], axis=0)
		input_matrix = jnp.hstack((tiled_tracking_robot_matrix, tiled_obstacle_points))

		input_matrix = (input_matrix - mu) / std

		A0 = jnp.maximum(0, self.W0 @ input_matrix.T + self.b0.T)
		A1 = jnp.maximum(0, self.W1 @ A0 + self.b1.T)  
		A2 = jnp.maximum(0, self.W2 @ A1 + self.b2.T)  
		occlusion_cost = (self.W3 @ A2 + self.b3.T)
		occlusion_cost = occlusion_cost

		occlusion_cost = occlusion_cost.reshape(self.ellite_num_projection, self.num, (obstacle_points.shape)[0])
		occlusion_cost = jnp.sum(occlusion_cost, axis=2)
		
		occlusion_cost = jnp.maximum(occlusion_cost, 0)
		occlusion_cost = jnp.sum(occlusion_cost, axis=1)


		### Rack Occlusion Computation
		tiled_tracking_trajectory_x_rack = jnp.tile(x_rack_traj, (self.ellite_num_projection))
		tiled_tracking_trajectory_y_rack = jnp.tile(y_rack_traj, (self.ellite_num_projection))

		tracking_robot_matrix_rack  = jnp.hstack((x_samples.reshape(self.ellite_num_projection * self.num, 1), y_samples.reshape(self.ellite_num_projection * self.num, 1), 
											tiled_tracking_trajectory_x_rack.reshape(self.ellite_num_projection * self.num, 1),
												tiled_tracking_trajectory_y_rack.reshape(self.ellite_num_projection * self.num, 1)))
		tiled_tracking_robot_matrix_rack = jnp.repeat(tracking_robot_matrix_rack, (obstacle_points.shape)[0], axis=0)
		input_matrix_rack = jnp.hstack((tiled_tracking_robot_matrix_rack, tiled_obstacle_points))

		input_matrix_rack = (input_matrix_rack - mu) / std

		A0 = jnp.maximum(0, self.W0 @ input_matrix_rack.T + self.b0.T)
		A1 = jnp.maximum(0, self.W1 @ A0 + self.b1.T)  
		A2 = jnp.maximum(0, self.W2 @ A1 + self.b2.T)  
		rack_occlusion_cost = (self.W3 @ A2 + self.b3.T)
		rack_occlusion_cost = rack_occlusion_cost

		rack_occlusion_cost = rack_occlusion_cost.reshape(self.ellite_num_projection, self.num, (obstacle_points.shape)[0])
		rack_occlusion_cost = jnp.sum(rack_occlusion_cost, axis=2)
		
		rack_occlusion_cost = jnp.maximum(rack_occlusion_cost, 0)
		rack_occlusion_cost = jnp.sum(rack_occlusion_cost, axis=1)

		cost_smoothness = ( jnp.linalg.norm(xddot_samples, axis = 1  )**2 +jnp.linalg.norm(yddot_samples, axis = 1  )**2 )

		total_cost = occlusion_cost * 1e4 + rack_occlusion_cost * 1e5 + cost_smoothness * 5 + 5 * res_norm_projection

		return total_cost

	@partial(jit,static_argnums=(0, ))
	def compute_ellite_projection(self, res_norm_batch, res_obs_norm, c_x_samples_vehicle, c_y_samples_vehicle,
										x_samples_vehicle, y_samples_vehicle, xdot_samples_vehicle, ydot_samples_vehicle, xddot_samples_vehicle, yddot_samples_vehicle):
		
		idx_ellite_projection = jnp.argsort(res_norm_batch)
		c_x_ellite_projection_vehicle = c_x_samples_vehicle[idx_ellite_projection[0:self.ellite_num_projection]]
		c_y_ellite_projection_vehicle = c_y_samples_vehicle[idx_ellite_projection[0:self.ellite_num_projection]]
		x_ellite_projection_vehicle = x_samples_vehicle[idx_ellite_projection[0:self.ellite_num_projection]]
		y_ellite_projection_vehicle = y_samples_vehicle[idx_ellite_projection[0:self.ellite_num_projection]]
		xdot_ellite_projection_vehicle = xdot_samples_vehicle[idx_ellite_projection[0:self.ellite_num_projection]]
		ydot_ellite_projection_vehicle = ydot_samples_vehicle[idx_ellite_projection[0:self.ellite_num_projection]]
		xddot_ellite_projection_vehicle = xddot_samples_vehicle[idx_ellite_projection[0:self.ellite_num_projection]]
		yddot_ellite_projection_vehicle = yddot_samples_vehicle[idx_ellite_projection[0:self.ellite_num_projection]]
		
		res_norm_projection = res_norm_batch[idx_ellite_projection[0:self.ellite_num_projection]]
		
		return c_x_ellite_projection_vehicle, c_y_ellite_projection_vehicle, x_ellite_projection_vehicle,\
			y_ellite_projection_vehicle, xdot_ellite_projection_vehicle, ydot_ellite_projection_vehicle, \
			xddot_ellite_projection_vehicle, yddot_ellite_projection_vehicle, res_norm_projection

	@partial(jit,static_argnums=(0, ) )
	def compute_ellite_samples(self, cost_batch, c_x_ellite_projection_vehicle, c_y_ellite_projection_vehicle,
										x_ellite_projection_vehicle, y_ellite_projection_vehicle, res_ellite_projection):
		
		idx_ellite = jnp.argsort(cost_batch)
		c_x_ellite_vehicle = c_x_ellite_projection_vehicle[idx_ellite[0:self.ellite_num]]
		c_y_ellite_vehicle = c_y_ellite_projection_vehicle[idx_ellite[0:self.ellite_num]]

		x_ellite_vehicle = x_ellite_projection_vehicle[idx_ellite[0:self.ellite_num]]
		y_ellite_vehicle = y_ellite_projection_vehicle[idx_ellite[0:self.ellite_num]]
		res_ellite = res_ellite_projection[idx_ellite[0:self.ellite_num]]

		return c_x_ellite_vehicle, c_y_ellite_vehicle, x_ellite_vehicle, y_ellite_vehicle, res_ellite

	@partial(jit,static_argnums=(0, ) )
	def compute_shifted_samples(self, c_x_ellite_vehicle, c_y_ellite_vehicle, x_obs_traj, y_obs_traj):

		key = random.PRNGKey(0)
		key, subkey = random.split(key)

		c_ellite = jnp.hstack(( c_x_ellite_vehicle, c_y_ellite_vehicle))
		c_mean = jnp.mean(c_ellite, axis = 0)
		c_cov = jnp.cov(c_ellite.T) + 0.01 * jnp.identity(2 * self.nvar)

		c_ellite_shift_temp = jax.random.multivariate_normal(key, c_mean, c_cov, (self.num_batch_projection, ))
		c_x_ellite_shift_vehicle = c_ellite_shift_temp[:, 0:self.nvar]
		c_y_ellite_shift_vehicle = c_ellite_shift_temp[:, self.nvar:2 * self.nvar]

		x_guess_vehicle_temp = jnp.dot(self.P_jax, c_x_ellite_shift_vehicle.T).T
		y_guess_vehicle_temp = jnp.dot(self.P_jax, c_y_ellite_shift_vehicle.T).T

		wc_alpha_temp_vehicle = (x_guess_vehicle_temp - x_obs_traj[:,jnp.newaxis])
		ws_alpha_temp_vehicle = (y_guess_vehicle_temp - y_obs_traj[:,jnp.newaxis])
		wc_alpha_vehicle = wc_alpha_temp_vehicle.transpose(1, 0, 2)
		ws_alpha_vehicle = ws_alpha_temp_vehicle.transpose(1, 0, 2)
		wc_alpha_vehicle = wc_alpha_vehicle.reshape(self.num_batch_projection , self.num * x_obs_traj.shape[0])
		ws_alpha_vehicle = ws_alpha_vehicle.reshape(self.num_batch_projection , self.num * y_obs_traj.shape[0])

		dist_obs_vehicle = -wc_alpha_vehicle**2/(self.a_obs**2) - ws_alpha_vehicle**2/(self.b_obs**2)+ 1
		dist_obs = dist_obs_vehicle

		cost_obs_penalty = jnp.linalg.norm(jnp.maximum(jnp.zeros((self.num_batch_projection, self.num * x_obs_traj.shape[0])), dist_obs), axis = 1)
		idx_ellite = jnp.argsort(cost_obs_penalty)

		c_x_ellite_shift_vehicle = c_x_ellite_shift_vehicle[idx_ellite[0:self.num_batch_projection-self.ellite_num]]
		c_y_ellite_shift_vehicle = c_y_ellite_shift_vehicle[idx_ellite[0:self.num_batch_projection-self.ellite_num]]

		c_x_bar = jnp.vstack((c_x_ellite_vehicle, c_x_ellite_shift_vehicle ))
		c_y_bar = jnp.vstack((c_y_ellite_vehicle, c_y_ellite_shift_vehicle))

		x_guess_vehicle = jnp.dot(self.P_jax, c_x_bar.T).T 
		y_guess_vehicle = jnp.dot(self.P_jax, c_y_bar.T).T 

		return c_x_bar, c_y_bar, x_guess_vehicle, y_guess_vehicle

	@partial(jit, static_argnums = (0,) )
	def get_closest_obstacles(self, obstacle_pointcloud, x_init, y_init):

		obstacle_distances = jnp.power(obstacle_pointcloud[:,0] - x_init, 2) + jnp.power(obstacle_pointcloud[:,1] - y_init, 2)
		points_with_min_dist = jnp.argsort(obstacle_distances)

		min_dist_pointcloud_x = obstacle_pointcloud[points_with_min_dist[0:self.num_obs],0]
		min_dist_pointcloud_y = obstacle_pointcloud[points_with_min_dist[0:self.num_obs],1]
		min_dist_pointcloud = jnp.vstack((min_dist_pointcloud_x, min_dist_pointcloud_y)).T

		return min_dist_pointcloud

    
	@partial(jit, static_argnums = (0,) )
	def compute_cem(self, key, x_init, vx_init, ax_init, y_init, vy_init, ay_init,
					 x_tracking, y_tracking, lamda_x, lamda_y, x_samples_init, y_samples_init, c_x_samples_init, c_y_samples_init,\
						xdot_samples_init, ydot_samples_init,\
                        xddot_samples_init, yddot_samples_init,
						vx_tracking, vy_tracking,
					   d_avg_tracking, obstacle_pointcloud, x_best_vehicle, y_best_vehicle,
					   x_rack_traj, y_rack_traj):
		
		x_obs_traj = jnp.repeat(obstacle_pointcloud[:,0], self.num).reshape(obstacle_pointcloud.shape[0],self.num)
		y_obs_traj = jnp.repeat(obstacle_pointcloud[:,1], self.num).reshape(obstacle_pointcloud.shape[0],self.num)

		for i in range(0, self.maxiter_cem):

			key, subkey = random.split(key)

			x_samples, y_samples, xdot_samples, ydot_samples, xddot_samples, yddot_samples, c_x_samples, c_y_samples, \
			alpha_v, d_v, alpha_a, d_a, alpha_tracking, d_tracking, lamda_x, lamda_y, \
				 res_norm_batch, res_obs_norm = self.compute_projection_samples(x_init, vx_init, ax_init, y_init, vy_init, ay_init
																,x_tracking, y_tracking, lamda_x, 
																lamda_y, x_samples_init, y_samples_init,\
																c_x_samples_init, c_y_samples_init, xdot_samples_init, ydot_samples_init,\
                        										xddot_samples_init, yddot_samples_init, vx_tracking, vy_tracking,\
																x_obs_traj, y_obs_traj, x_best_vehicle, y_best_vehicle)
			c_x_ellite_projection, c_y_ellite_projection,\
			x_ellite_projection, y_ellite_projection, xdot_ellite_projection, ydot_ellite_projection,\
			xddot_ellite_projection, yddot_ellite_projection,\
			res_norm_ellite_projection = self.compute_ellite_projection(res_norm_batch, res_obs_norm,c_x_samples, c_y_samples, \
																x_samples, y_samples, xdot_samples, ydot_samples, xddot_samples, yddot_samples )

			cost_batch = self.compute_cost_batch(xddot_ellite_projection, yddot_ellite_projection, x_ellite_projection, y_ellite_projection,
													 d_avg_tracking, x_rack_traj, y_rack_traj, obstacle_pointcloud, res_norm_ellite_projection,
													 x_best_vehicle, y_best_vehicle)
			c_x_ellite, c_y_ellite,\
			x_ellite, y_ellite, res_ellite = self.compute_ellite_samples(cost_batch, c_x_ellite_projection, c_y_ellite_projection,\
																			x_ellite_projection, y_ellite_projection, res_norm_ellite_projection)

			c_x_samples_init, c_y_samples_init,\
			x_samples_init, y_samples_init= self.compute_shifted_samples(c_x_ellite, c_y_ellite, x_obs_traj, y_obs_traj)

		c_x_best = c_x_ellite[0]
		c_y_best = c_y_ellite[0]
		x_best = x_ellite[0]
		y_best = y_ellite[0]


		return c_x_best, c_y_best, x_best, y_best, c_x_ellite_projection, c_y_ellite_projection,\
				x_ellite_projection, y_ellite_projection

	@partial(jit, static_argnums = (0,) )
	def compute_controls(self, c_x_best, c_y_best, dt_up, vx_tracking, vy_tracking, 
							t_update, tot_time_copy_up, x_init, y_init, alpha_init,
                            x_tracking_init, y_tracking_init):
		
		num_average_samples = 10
		x_up = jnp.dot(self.P_up_jax, c_x_best)
		y_up = jnp.dot(self.P_up_jax, c_y_best)
		
		xddot_up = jnp.dot(self.Pddot_up_jax, c_x_best)
		yddot_up = jnp.dot(self.Pddot_up_jax, c_y_best)

		xdot_up = jnp.dot(self.Pdot_up_jax, c_x_best)
		ydot_up = jnp.dot(self.Pdot_up_jax, c_y_best)
		
		vx_control = jnp.mean(xdot_up[0:num_average_samples])
		vy_control = jnp.mean(ydot_up[0:num_average_samples])

		ax_control = jnp.mean(xddot_up[0:num_average_samples])
		ay_control = jnp.mean(yddot_up[0:num_average_samples])


		vx_local = xdot_up * jnp.cos(alpha_init) + ydot_up * jnp.sin(alpha_init)
		vy_local = -xdot_up * jnp.sin(alpha_init) + ydot_up * jnp.cos(alpha_init)

		vx_control_local = jnp.mean(vx_local[0:num_average_samples])
		vy_control_local = jnp.mean(vy_local[0:num_average_samples])

		alphadot = 0
		alphadot_drone = 0

		return vx_control_local, vy_control_local, ax_control, ay_control, \
						 alphadot_drone, jnp.mean(x_up[0:num_average_samples]), jnp.mean(y_up[0:num_average_samples]), vx_control, vy_control
		





