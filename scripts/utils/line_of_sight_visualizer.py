#!/usr/bin/env python

import rospy
# import math
import tf
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

from numpy import *
import scipy.special
from scipy import interpolate
from scipy.io import loadmat
import random
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseStamped
import tf
import message_filters
from nav_msgs.msg import Odometry
from nav_msgs.msg import Path

line_of_sight_marker = Path()
line_of_sight_marker.header.frame_id = "los"

def odom_broadcaster(drone_odom):
  br = tf.TransformBroadcaster()
  br.sendTransform((drone_odom.pose.pose.position.x, drone_odom.pose.pose.position.y, drone_odom.pose.pose.position.z)
  ,(drone_odom.pose.pose.orientation.x, drone_odom.pose.pose.orientation.y,
   drone_odom.pose.pose.orientation.z, drone_odom.pose.pose.orientation.w),
                   rospy.Time.now(),"base_link","world")

def odomCallback(drone_odom, target_odom):
  global line_of_sight_marker, line_of_sight_pub
  line_sight_temp_point_1 = PoseStamped()
  line_sight_temp_point_2 = PoseStamped()
  line_of_sight_marker.poses=[]

  line_sight_temp_point_1.pose.position.x = drone_odom.pose.pose.position.x
  line_sight_temp_point_1.pose.position.y = drone_odom.pose.pose.position.y
  line_sight_temp_point_1.pose.position.z = 2.5

  line_sight_temp_point_2.pose.position.x = target_odom.pose.pose.position.x
  line_sight_temp_point_2.pose.position.y = target_odom.pose.pose.position.y
  line_sight_temp_point_2.pose.position.z = 0

  line_of_sight_marker.poses = [line_sight_temp_point_1,line_sight_temp_point_2]
  line_of_sight_marker.header.stamp = rospy.Time.now()

  line_of_sight_pub.publish(line_of_sight_marker)

if __name__ == '__main__':

  rospy.init_node('los_visualizer')
  traj_number = 8
  drone_odom_sub = message_filters.Subscriber('/bebop/odom', Odometry)
  target_odom_sub = message_filters.Subscriber('/robotont/odom', Odometry)
  
  ts = message_filters.ApproximateTimeSynchronizer([drone_odom_sub, target_odom_sub], 1,1)
  ts.registerCallback(odomCallback)
  line_of_sight_pub = rospy.Publisher( 'line_of_sight', Path, queue_size=100)
  rospy.spin()

  




