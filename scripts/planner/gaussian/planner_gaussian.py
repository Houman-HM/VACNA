#!/usr/bin/python
import vehicle_module_gaussian as vehicle_module
import drone_module_gaussian as drone_module
import numpy as np
import jax.numpy as jnp
import jax
import bernstein_coeff_order10_arbitinterval
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy
import time
from jax import vmap, random
import rospy
import rospkg
import threading
from scipy.io import loadmat
from nav_msgs.msg import Odometry
import sys
from geometry_msgs.msg import Twist
from geometry_msgs.msg import TwistStamped
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import PointCloud

from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import PointField
from std_msgs.msg import Header
from sensor_msgs import point_cloud2
import message_filters
from nav_msgs.msg import Odometry
import open3d
from std_srvs.srv import Empty
import copy

robot_cmd_publisher = None
drone_pose_vel = []
vehicle_pose_vel = []

is_received = False
drone_cmd_publisher = None
vehicle_cmd_publisher = None

robot_traj_marker_publisher = None
robot_traj_publisher = None
obstacle_points = None


is_received = False
trajectory_updated = False
robot_cmd_publisher = None
robot_traj_marker_publisher = None
robot_traj_publisher = None
robot_traj_publisher = None
pointcloud_publisher = None

obstacle_points_drone = None
obstacle_points_vehicle = None

min_dis_points_vehicle, min_dis_points_drone = None, None

pointcloud_mutex = threading.Lock()
odom_mutex = threading.Lock()
publish_traj_mutex = threading.Lock()

num_laser_points = 720
num_down_sampled = 200
x_obs_pointcloud_drone = np.ones((num_laser_points,1)) * 100
y_obs_pointcloud_drone = np.ones((num_laser_points,1)) * 100

x_obs_pointcloud_vehicle = np.ones((num_laser_points,1)) * 100
y_obs_pointcloud_vehicle = np.ones((num_laser_points,1)) * 100

x_obs_down_sampled = np.ones((num_down_sampled,1)) * 100
y_obs_down_sampled = np.ones((num_down_sampled,1)) * 100
xyz = np.random.rand(720, 3)

line_to_track_publisher = None
line_to_track_marker = None
drone_traj_marker, vehicle_traj_marker, drone_traj_publisher, vehicle_traj_publisher = None, None, None, None

x_drone_traj_global, y_drone_traj_global, x_vehicle_traj_global, y_vehicle_traj_global, x_tracking_traj_global, y_tracking_traj_global = None, None, None, None, None, None


def createLineMarker(id = 1, frame_id = 'base', scale = [0.1,0,0], color = [1,0,0,1]):

    marker = Marker() 
    marker.id = id
    marker.header.frame_id = frame_id
    marker.type = Marker.LINE_STRIP
    marker.action = Marker.ADD
    marker.scale.x = scale[0]
    marker.scale.y = scale[1]
    marker.scale.z = scale[2]
    marker.pose.orientation.w = 1.0
    marker.color.a = color[0]
    marker.color.r = color[1]
    marker.color.g = color[2]
    marker.color.b = color[3]
    
    return marker

def odomCallback(vehicle_odom, drone_odom):

    global is_received, target_pose_vel, drone_pose_vel, vehicle_pose_vel, odom_mutex
    odom_mutex.acquire()

    drone_orientation_q = drone_odom.pose.pose.orientation
    drone_orientation_list = [drone_orientation_q.x, drone_orientation_q.y, drone_orientation_q.z, drone_orientation_q.w]

    vehicle_orientation_q = vehicle_odom.pose.pose.orientation
    vehicle_orientation_list = [vehicle_orientation_q.x, vehicle_orientation_q.y, vehicle_orientation_q.z, vehicle_orientation_q.w]

    (drone_roll, drone_pitch, drone_yaw) = euler_from_quaternion (drone_orientation_list)
    (vehicle_roll, vehicle_pitch, vehicle_yaw) = euler_from_quaternion (vehicle_orientation_list)

    drone_pose_vel = [drone_odom.pose.pose.position.x, drone_odom.pose.pose.position.y, drone_odom.pose.pose.position.z ,drone_yaw, 
                    drone_odom.twist.twist.linear.x, drone_odom.twist.twist.linear.y, drone_odom.twist.twist.linear.z, drone_odom.twist.twist.angular.z]

    vehicle_pose_vel = [vehicle_odom.pose.pose.position.x, vehicle_odom.pose.pose.position.y, vehicle_odom.pose.pose.position.z ,vehicle_yaw, 
                    vehicle_odom.twist.twist.linear.x, vehicle_odom.twist.twist.linear.y, vehicle_odom.twist.twist.linear.z, vehicle_odom.twist.twist.angular.z]
    odom_mutex.release()


def pointcloudCallback(vehicle_msg):
    global x_obs_pointcloud_drone, y_obs_pointcloud_drone,\
            x_obs_pointcloud_vehicle, y_obs_pointcloud_vehicle,\
            is_received, pointcloud_mutex, obstacle_points_drone, obstacle_points_vehicle, x_obs_down_sampled, y_obs_down_sampled, xyz,\
            obstacle_points_main, odom_mutex
    
    msg_len= len(vehicle_msg.points)
    pointcloud_mutex.acquire()
    increment_value = 1
    inner_counter = 0
    start_time = time.time()
    for nn in range(0,msg_len, increment_value):
        x_obs_pointcloud_vehicle[inner_counter] = vehicle_msg.points[nn].x
        y_obs_pointcloud_vehicle[inner_counter] = vehicle_msg.points[nn].y
        inner_counter+=1
    
    idxes = np.argwhere((x_obs_pointcloud_vehicle[:]>=80) | (y_obs_pointcloud_vehicle[:]>=80))
    x_obs_pointcloud_vehicle[idxes] = x_obs_pointcloud_vehicle[0]
    y_obs_pointcloud_vehicle[idxes] = y_obs_pointcloud_vehicle[0]
    
    xyz[:,0] = x_obs_pointcloud_vehicle.flatten()
    xyz[:,1] = y_obs_pointcloud_vehicle.flatten()
    xyz[:,2] = 1
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(xyz)
    downpcd = pcd.voxel_down_sample(voxel_size=0.8)
    downpcd_array  = np.asarray(downpcd.points)
    num_down_sampled_points = downpcd_array[:,0].shape[0]
    x_obs_down_sampled = np.ones((200,1)) * 100
    y_obs_down_sampled = np.ones((200,1)) * 100
    x_obs_down_sampled[0:num_down_sampled_points,0]= downpcd_array[:,0]
    y_obs_down_sampled[0:num_down_sampled_points,0] = downpcd_array[:,1]
    x_obs_down_sampled[num_down_sampled_points-1:]= 1000
    y_obs_down_sampled[num_down_sampled_points-1:]= 1000
    
    obstacle_points_vehicle = np.hstack((x_obs_down_sampled,y_obs_down_sampled))
    
    pointcloud_mutex.release()
    is_received = True
    inner_counter = 0


def publishTrajectories():

    global trajectory_updated, drone_traj_marker, vehicle_traj_marker, drone_traj_publisher, vehicle_traj_publisher,\
        vehicle_traj_publisher, line_to_track_marker, line_to_track_publisher, \
        x_drone_traj_global, y_drone_traj_global, x_vehicle_traj_global, y_vehicle_traj_global,\
        x_tracking_traj_global, y_tracking_traj_global
    
    while (True):
        if (trajectory_updated):
            publish_traj_mutex.acquire()
            x_drone_traj, y_drone_traj, x_vehicle_traj, y_vehicle_traj, x_tracking_traj, y_tracking_traj = x_drone_traj_global, y_drone_traj_global,\
                x_vehicle_traj_global, y_vehicle_traj_global, x_tracking_traj_global, y_tracking_traj_global
            publish_traj_mutex.release()

            for i in range (x_drone_traj.shape[0]):
                drone_point = Point()
                vehicle_point = Point()
                tracking_point = Point()

                drone_point.x = x_drone_traj[i]
                drone_point.y = y_drone_traj[i]
                vehicle_point.x = x_vehicle_traj[i]
                vehicle_point.y = y_vehicle_traj[i]
                tracking_point.x = x_tracking_traj[0,i]
                tracking_point.y = y_tracking_traj[0,i]
                line_to_track_marker.points.append(tracking_point)
                if (i<=30):
                    drone_traj_marker.points.append(drone_point)
                    vehicle_traj_marker.points.append(vehicle_point)

            
            drone_traj_publisher.publish(drone_traj_marker)
            vehicle_traj_publisher.publish(vehicle_traj_marker)
            line_to_track_publisher.publish(line_to_track_marker)
            rospy.sleep(0.001)
            
            drone_traj_marker.points = []
            vehicle_traj_marker.points = []
            line_to_track_marker.points = []

        rospy.sleep(0.001)


def mpc():

    global is_received, trajectory_updated, drone_cmd_publisher, vehicle_cmd_publisher,\
    robot_traj_publisher, robot_traj_marker_publisher, obstacle_points_vehicle,\
    x_drone_traj_global, y_drone_traj_global, x_vehicle_traj_global, y_vehicle_traj_global,\
    x_tracking_traj_global, y_tracking_traj_global, drone_pose_vel, vehicle_pose_vel,\
    min_dis_points_vehicle, min_dis_points_drone,odom_mutex, pointcloud_mutex
    
    rospack = rospkg.RosPack()
    package_path = rospack.get_path("vacna")
    v_max = 1
    a_max = 1.5
    num_batch_projection = 100

    rho_ineq = 1
    rho_projection = 1
    rho_tracking = 0.8
    rho_obs = 10
    maxiter_projection = 3
    maxiter_cem = 6
    maxiter_mpc = 15000
    occlusion_weight = 1

    ellite_num_projection =  100
    ellite_num = 20

    d_min_tracking = 0.2
    d_max_tracking = 0.9


    d_avg_tracking = (d_min_tracking+d_max_tracking)/2.0

    initial_up_sampling = 20

    target_distance_weight = 0.1
    smoothness_weight = 100

    a_obs = 0.53
    b_obs = 0.53
    num_obs = 20

    ### Drone parameters

    v_max_drone = 1
    a_max_drone = 1.5
    num_batch_projection_drone = 100

    rho_ineq_drone = 1
    rho_projection_drone = 1
    rho_tracking_drone = 0.8
    rho_obs_drone = 1
    maxiter_projection_drone = 3
    maxiter_cem_drone = 6
    occlusion_weight_drone = 1e+4

    ellite_num_projection_drone =  100
    ellite_num_drone = 20

    d_min_tracking_drone = 0.0
    d_max_tracking_drone = 1

    d_min_tracking_vehicle = 0
    d_max_tracking_vehicle = 0.7

    initial_up_sampling_drone = 20

    target_distance_weight_drone = 0.1
    smoothness_weight_drone = 50

    a_obs_drone = 0.53
    b_obs_drone = 0.53
    num_obs_drone = 20

    ############# parameters

    t_fin = 10.0
    num = 100
    tot_time = np.linspace(0, t_fin, num)
    tot_time_copy = tot_time.reshape(num, 1)
            
    P, Pdot, Pddot = bernstein_coeff_order10_arbitinterval.bernstein_coeff_order10_new(10, tot_time_copy[0], tot_time_copy[-1], tot_time_copy)
    nvar = np.shape(P)[1]

    ###################################
    t_update = 0.05
    num_up = 200
    dt_up = t_fin/num_up
    tot_time_up = np.linspace(0, t_fin, num_up)
    tot_time_copy_up = tot_time_up.reshape(num_up, 1)

    P_up, Pdot_up, Pddot_up = bernstein_coeff_order10_arbitinterval.bernstein_coeff_order10_new(10, tot_time_copy_up[0], tot_time_copy_up[-1], tot_time_copy_up)

    P_up_jax = jnp.asarray(P_up)
    Pdot_up_jax = jnp.asarray(Pdot_up)
    Pddot_up_jax = jnp.asarray(Pddot_up)

    ########################################

    x_init_drone =  0
    vx_init_drone = 0.0
    ax_init_drone = 0.0

    y_init_drone =  0
    vy_init_drone = 0.0
    ay_init_drone = 0.0

    x_init_vehicle =  0
    vx_init_vehicle = 0.0
    ax_init_vehicle = 0.0

    y_init_vehicle =  0
    vy_init_vehicle = 0.0
    ay_init_vehicle = 0.0

    ##############################################################

    prob = vehicle_module.batch_occ_tracking(P, Pdot, Pddot, v_max, a_max, t_fin, num, num_batch_projection,
                                                tot_time, rho_ineq, maxiter_projection, rho_projection, 
                                                rho_tracking, rho_obs, maxiter_cem, d_min_tracking, d_max_tracking,
                                                P_up_jax, Pdot_up_jax, Pddot_up_jax, occlusion_weight,
                                                ellite_num_projection, ellite_num,
                                                initial_up_sampling, target_distance_weight, smoothness_weight, a_obs, b_obs, num_obs)

    prob_drone = drone_module.batch_occ_tracking(P, Pdot, Pddot, v_max_drone, a_max_drone, t_fin, num, num_batch_projection_drone,
                                                tot_time, rho_ineq_drone, maxiter_projection_drone, rho_projection_drone, 
                                                rho_tracking_drone, rho_obs_drone, maxiter_cem_drone, d_min_tracking_drone, d_max_tracking_drone,
                                                P_up_jax, Pdot_up_jax, Pddot_up_jax, occlusion_weight_drone,
                                                ellite_num_projection_drone, ellite_num_drone,
                                                initial_up_sampling_drone, target_distance_weight_drone, smoothness_weight_drone,
                                                 a_obs_drone, b_obs_drone, num_obs_drone, d_min_tracking_vehicle, d_max_tracking_vehicle)

    weight_biases_mat_file = loadmat(package_path + "/nn_weights/occlusion_model/nn_weight_biases.mat")

    prob.W0, prob.b0, prob.W1, \
    prob.b1, prob.W2, prob.b2, \
    prob.W3, prob.b3 = vehicle_module.get_weights_biases(weight_biases_mat_file)


    prob_drone.W0, prob_drone.b0, prob_drone.W1, \
    prob_drone.b1, prob_drone.W2, prob_drone.b2, \
    prob_drone.W3, prob_drone.b3 = drone_module.get_weights_biases(weight_biases_mat_file)

    lamda_x_drone = jnp.zeros((num_batch_projection, nvar))
    lamda_y_drone = jnp.zeros((num_batch_projection, nvar))

    lamda_x_vehicle = jnp.zeros((num_batch_projection, nvar))
    lamda_y_vehicle = jnp.zeros((num_batch_projection, nvar))		

    key = random.PRNGKey(0)
    odom_mutex.acquire()

    x_init_drone = drone_pose_vel[0] - vehicle_pose_vel[0]
    y_init_drone = drone_pose_vel[1] - vehicle_pose_vel[1]

    x_init_vehicle = 0
    y_init_vehicle = 0

    odom_mutex.release()

     # Example 1
    x_points_track = [5.5, 5.5, 0.2, 0.2, -6, -6, -11.9, -11.9]
    y_points_track = [-33, 33, 33, -33, -33, 33, 33, -33]
    
    rack_x_points = [4.8, 4.8, -1.4, -1.4, -7, -7 , -12.9, -12.9]
    rack_y_points = [-33, 33, 33, -33, -33, 33, 33, -33]
 
    vx = 0
    vy = 10
    tracking_line_resolution = 5000
    tracking_time = np.linspace(0, t_fin, tracking_line_resolution)
    x_track_init = x_points_track[0]
    y_track_init = y_points_track[0]

    x_best_vehicle = x_init_vehicle + 0 * prob.tot_time
    y_best_vehicle = y_init_vehicle + 1 * prob.tot_time
    
    counter = 0
    rospy.loginfo("The planner started!")
    rospy.loginfo("Waiting for the initial JAX compilation!")
    for i in range(0, maxiter_mpc):
        start_time = time.time()
        pointcloud_mutex.acquire()
        odom_mutex.acquire()

        vehicle_world_pose_x = copy.deepcopy(vehicle_pose_vel[0])
        vehicle_world_pose_y = copy.deepcopy(vehicle_pose_vel[1]) 
        obstacles = copy.deepcopy(obstacle_points_vehicle)
        obstacles[:,0] = obstacles[:,0] - vehicle_world_pose_x
        obstacles[:,1] = obstacles[:,1] - vehicle_world_pose_y
        vehicle_jax_obstacle_points = jnp.asarray(obstacles)

        vx_tracking = vehicle_pose_vel[4] * np.cos(vehicle_pose_vel[3])
        vy_tracking = vehicle_pose_vel[4] * np.sin(vehicle_pose_vel[3])
        
        drone_alpha_init = drone_pose_vel[3]
        vehicle_alpha_init = vehicle_pose_vel[3]
        
        distance_from_goal = np.sqrt((x_points_track[counter+1] - (x_init_drone + vehicle_world_pose_x))**2 + (y_points_track[counter+1] - (y_init_drone + vehicle_world_pose_y))**2)
        if (distance_from_goal <=2) and counter<6:
            counter+=1
        if counter == 0 or counter ==4:
            vx= 0
            vy = 0.8
        elif (counter %2):
            vx=-0.8
            vy = 0
        else:
            vx=0
            vy=-0.8
   
        tracking_line_resolution = 20000
        tracking_time = np.linspace(0, t_fin, tracking_line_resolution)
        
        x_track_init = x_points_track[counter] - vehicle_world_pose_x
        y_track_init = y_points_track[counter] - vehicle_world_pose_y
        x_tracking_rack = rack_x_points[counter]- vehicle_world_pose_x
        y_tracking_rack = rack_y_points[counter]- vehicle_world_pose_y

        x_global_tracking = (x_track_init + 20 * vx * tracking_time)
        y_global_tracking = (y_track_init + 20 * vy * tracking_time)

        x_global_tracking_rack = (x_tracking_rack + 20 * vx * tracking_time)
        y_global_tracking_rack = (y_tracking_rack + 20 * vy * tracking_time)


        projected_points_on_tracking_line = (x_global_tracking  - x_init_drone)**2 + (y_global_tracking - y_init_drone)**2
        min_distance_index = jnp.argmin(projected_points_on_tracking_line)
        x_init_tracking = x_global_tracking[min_distance_index]
        y_init_tracking = y_global_tracking[min_distance_index]

        x_tracking_drone = (x_init_tracking + vx * prob.tot_time).reshape(1, num)
        y_tracking_drone = (y_init_tracking + vy * prob.tot_time).reshape(1, num)


        projected_points_on_tracking_line_rack = (x_global_tracking_rack  - x_init_drone)**2 + (y_global_tracking_rack - y_init_drone)**2
        min_distance_index_rack = jnp.argmin(projected_points_on_tracking_line_rack)
        x_init_tracking_rack = x_global_tracking_rack[min_distance_index_rack]
        y_init_tracking_rack = y_global_tracking_rack[min_distance_index_rack]


        x_tracking_drone = (x_init_tracking + vx * prob.tot_time).reshape(1, num)
        y_tracking_drone = (y_init_tracking + vy * prob.tot_time).reshape(1, num)

        x_rack_traj = (x_init_tracking_rack + vx * prob.tot_time).reshape(1, num)
        y_rack_traj = (y_init_tracking_rack + vy * prob.tot_time).reshape(1, num)
        
        x_fin_drone = x_tracking_drone[0, -1]
        y_fin_drone = y_tracking_drone[0, -1]

        drone_pose_vel_temp  = copy.deepcopy(np.copy(drone_pose_vel))
        vehicle_pose_vel_temp = copy.deepcopy(np.copy(vehicle_pose_vel))
        drone_pose_vel_temp[0] = drone_pose_vel_temp[0] - vehicle_pose_vel_temp[0]
        drone_pose_vel_temp[1] = drone_pose_vel_temp[1] - vehicle_pose_vel_temp[1]

        pointcloud_mutex.release()
        odom_mutex.release()

        
        
        vehicle_jax_obstacle_points_min = prob.get_closest_obstacles(vehicle_jax_obstacle_points, x_init_vehicle, y_init_vehicle)
        drone_jax_obstacle_points = prob_drone.get_closest_obstacles(vehicle_jax_obstacle_points, x_init_drone, y_init_drone)
        
        c_x_samples_init_drone, c_y_samples_init_drone, x_samples_init_drone, y_samples_init_drone,\
        xdot_samples_init_drone, ydot_samples_init_drone,\
        xddot_samples_init_drone, yddot_samples_init_drone = prob_drone.compute_initial_samples(x_init_drone, y_init_drone, x_fin_drone, y_fin_drone, drone_jax_obstacle_points)
        
        c_x_best_drone, c_y_best_drone, x_best_drone, y_best_drone, c_x_ellite_projection_drone, c_y_ellite_projection_drone,\
        x_ellite_projection_drone, y_ellite_projection_drone = prob_drone.compute_cem(key, x_init_drone, vx_init_drone, ax_init_drone, y_init_drone, vy_init_drone, ay_init_drone,
                                                    x_tracking_drone, y_tracking_drone, lamda_x_drone, lamda_y_drone, x_samples_init_drone, y_samples_init_drone, c_x_samples_init_drone, c_y_samples_init_drone,\
                                                    xdot_samples_init_drone, ydot_samples_init_drone,\
                                                    xddot_samples_init_drone, yddot_samples_init_drone, vx_tracking, vy_tracking,
                                                    d_avg_tracking, drone_jax_obstacle_points, x_best_vehicle.reshape(1, num),
                                                     y_best_vehicle.reshape(1, num), x_rack_traj, y_rack_traj)
                                                     
        c_x_samples_init_vehicle, c_y_samples_init_vehicle, x_samples_init_vehicle, y_samples_init_vehicle,\
        xdot_samples_init_vehicle, ydot_samples_init_vehicle,\
        xddot_samples_init_vehicle, yddot_samples_init_vehicle = prob.compute_initial_samples(x_init_vehicle, y_init_vehicle, x_best_drone[-1], y_best_drone[-1],
                                                                                             vehicle_jax_obstacle_points)

        c_x_best_vehicle, c_y_best_vehicle, x_best_vehicle, y_best_vehicle, c_x_ellite_projection_vehicle, c_y_ellite_projection_vehicle,\
        x_ellite_projection_vehicle, y_ellite_projection_vehicle = prob.compute_cem(key, x_init_vehicle, vx_init_vehicle, ax_init_vehicle, y_init_vehicle, vy_init_vehicle, ay_init_vehicle,
                                                    x_best_drone.reshape(1, num), y_best_drone.reshape(1, num), lamda_x_vehicle, lamda_y_vehicle, x_samples_init_vehicle, y_samples_init_vehicle, c_x_samples_init_vehicle, c_y_samples_init_vehicle,\
                                                    xdot_samples_init_vehicle, ydot_samples_init_vehicle,\
                                                    xddot_samples_init_vehicle, yddot_samples_init_vehicle, vx_tracking, vy_tracking,
                                                    d_avg_tracking, vehicle_jax_obstacle_points_min)
        


        vx_control_local_vehicle, vy_control_local_vehicle, ax_control_vehicle, \
        ay_control_vehicle, vangular_control_vehicle, robot_traj_x_vehicle, robot_traj_y_vehicle,\
        vx_control_vehicle, vy_control_vehicle= prob.compute_controls(c_x_best_vehicle, c_y_best_vehicle, dt_up, vx_tracking, vy_tracking, 
                                                                                t_update, tot_time_copy_up, x_init_vehicle, y_init_vehicle, vehicle_alpha_init,
                                                                                    x_tracking_drone, y_tracking_drone)
        vx_control_local_vehicle, vy_control_local_vehicle, ax_control_vehicle, \
        ay_control_vehicle, vangular_control_vehicle,\
        vx_control_vehicle, vy_control_vehicle = np.asarray(vx_control_local_vehicle), np.asarray(vy_control_local_vehicle), np.asarray(ax_control_vehicle),\
                                                np.asarray(ay_control_vehicle), np.asarray(vangular_control_vehicle), np.asarray(vx_control_vehicle),\
                                                np.asarray(vy_control_vehicle)

        vx_control_local_drone, vy_control_local_drone, ax_control_drone, \
        ay_control_drone, vangular_control_drone, robot_traj_x_drone, robot_traj_y_drone,\
        vx_control_drone, vy_control_drone= prob_drone.compute_controls(c_x_best_drone, c_y_best_drone, dt_up, vx_tracking, vy_tracking, 
                                                                                t_update, tot_time_copy_up, x_init_drone, y_init_drone, drone_alpha_init,
                                                                                    x_tracking_drone, y_tracking_drone)

        vx_control_local_drone, vy_control_local_drone, ax_control_drone, \
        ay_control_drone, vangular_control_drone,\
        vx_control_drone, vy_control_drone = np.asarray(vx_control_local_drone), np.asarray(vy_control_local_drone), np.asarray(ax_control_drone),\
                                                np.asarray(ay_control_drone), np.asarray(vangular_control_drone), np.asarray(vx_control_drone),\
                                                np.asarray(vy_control_drone)

        
        publish_traj_mutex.acquire()

        x_best_vehicle_temp = np.asarray(x_best_vehicle) + vehicle_world_pose_x
        y_best_vehicle_temp = np.asarray(y_best_vehicle) + vehicle_world_pose_y
        x_best_drone_temp = np.asarray(x_best_drone) + vehicle_world_pose_x
        y_best_drone_temp = np.asarray(y_best_drone) + vehicle_world_pose_y

        x_tracking_drone_temp = np.asarray(x_tracking_drone) + vehicle_world_pose_x
        y_tracking_drone_temp  = np.asarray(y_tracking_drone) + vehicle_world_pose_y

        x_drone_traj_global, y_drone_traj_global, x_vehicle_traj_global, y_vehicle_traj_global,\
        x_tracking_traj_global, y_tracking_traj_global, = x_best_drone_temp, y_best_drone_temp,\
                                                        x_best_vehicle_temp, y_best_vehicle_temp,\
                                                        x_tracking_drone_temp, y_tracking_drone_temp

        trajectory_updated = True
        publish_traj_mutex.release()
        
        cmd_drone = Twist()
        cmd_vehicle = Twist()
        if (not jnp.isnan(vx_control_local_vehicle) or not jnp.isnan(vx_control_local_vehicle)):

            cmd_drone.linear.x= vx_control_local_drone
            cmd_drone.linear.y= vy_control_local_drone
            cmd_drone.angular.z= vangular_control_drone

            cmd_vehicle.linear.x= vx_control_local_vehicle
            cmd_vehicle.linear.y= vy_control_local_vehicle
            cmd_vehicle.angular.z= vangular_control_vehicle

        else:
            cmd_drone.linear.x= 0
            cmd_drone.linear.y= 0
            cmd_vehicle.linear.x= 0
            cmd_vehicle.linear.y= 0
        if (i>2):
            
            drone_cmd_publisher.publish(cmd_drone)
            vehicle_cmd_publisher.publish(cmd_vehicle)
        odom_mutex.acquire()

        x_init_drone = drone_pose_vel[0] - vehicle_pose_vel[0] 
        y_init_drone = drone_pose_vel[1] - vehicle_pose_vel[1]

        x_init_vehicle = 0
        y_init_vehicle = 0

        odom_mutex.release()

        vx_init_drone = vx_control_drone
        vy_init_drone = vy_control_drone
        ax_init_drone = ax_control_drone
        ay_init_drone = ay_control_drone

        vx_init_vehicle = vx_control_vehicle
        vy_init_vehicle = vy_control_vehicle
        ax_init_vehicle = ax_control_vehicle
        ay_init_vehicle = ay_control_vehicle
        time_taken = time.time() - start_time
        rospy.loginfo ("Time taken: %s", str(time_taken))
        rospy.sleep(0.00001)


if __name__ == "__main__":
    
    rospy.init_node('planner_gaussian')
    drone_traj_marker = createLineMarker(color=[1,1,0,0])
    vehicle_traj_marker = createLineMarker(color=[1,0,0,1])
    line_to_track_marker = createLineMarker(color=[1,0,0,0])

    drone_cmd_publisher = rospy.Publisher('bebop/cmd_vel', Twist, queue_size=10)
    vehicle_cmd_publisher = rospy.Publisher('robotont/cmd_vel', Twist, queue_size=10)

    drone_traj_publisher = rospy.Publisher('/drone_traj', Marker, queue_size=10)
    vehicle_traj_publisher = rospy.Publisher('/vehicle_traj', Marker, queue_size=10)

    line_to_track_publisher = rospy.Publisher('/line_to_track', Marker, queue_size=10)

    vehicle_pointcloud_sub = rospy.Subscriber('pointcloud', PointCloud, pointcloudCallback)

    drone_odom_sub = message_filters.Subscriber('/bebop/odom', Odometry)
    vehicle_odom_sub = message_filters.Subscriber('/robotont/odom', Odometry)

    pointcloud_publisher_vehicle = rospy.Publisher('/vehicle_pointcloud', Marker, queue_size=10)
    pointcloud_publisher_drone = rospy.Publisher('/drone_pointcloud', Marker, queue_size=10)

    ts = message_filters.ApproximateTimeSynchronizer([vehicle_odom_sub, drone_odom_sub], 1,1, allow_headerless=True)
    ts.registerCallback(odomCallback)

    mpc_thread = threading.Thread(target=mpc)
    trajectory_publisher_thread = threading.Thread(target=publishTrajectories)
    mpc_thread.start()
    trajectory_publisher_thread.start()
    rospy.spin()





