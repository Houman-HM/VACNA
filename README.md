# VACNA
Ths repository is associated with our RAL submission paper: "VACNA: Visibility-Aware Cooperative Planning with Applcation in Inventory Management"

# Dependencies:

* [JAX](https://github.com/google/jax)
* [bebop_simulator](https://github.com/Houman-HM/bebop_simulator/tree/bebop_hokuyo)
* [robotont_gazebo](https://github.com/robotont/robotont_gazebo)
* [gazebo2rviz](https://github.com/andreasBihlmaier/gazebo2rviz) (If you need the RViz visualization)

## Demo Video
[![Watch the video](https://img.youtube.com/vi/jXQJUyfzIzU/maxresdefault.jpg)](https://youtu.be/jXQJUyfzIzU)

## Installation procedure
After installing the dependencies, you can build our propsed MPC package as follows:
``` 
cd your_catkin_ws/src
git clone https://github.com/Houman-HM/VACNA
cd .. && catkin build
source your_catkin_ws/devel/setup.bash
```
## Running the algorithm

In order to run the MPC for the warehouse inventory, follow the procedure below:

### In the first terminal:
```
roslaunch vacna warehouse.launch
```

This launches a warehouse environment in Gazebo.
### In the second terminal:

#### For the MPC with Gaussian Initialization:

```
rosrun vacna planner_gaussian.py
```
#### For the MPC with CVAE Initialization:
```
rosrun vacna planner_cvae.py
```

## Training the CVAE Model

![HMNet Pipeline_page-0001](https://github.com/jatan12/MPC-Bi-Level/assets/38403732/b1fda1a8-c7cd-4e86-9ea5-13ee0e4fa50b)

1. Clone this repository:

```
git clone https://github.com/Houman-HM/VACNA.git
cd VACNA
```
2. Create a conda environment and install the dependencies:

```
conda create -n vacna python=3.8
conda activate vacna
pip install -r requirements.txt
```
3. Download the [training dataset](https://owncloud.ut.ee/owncloud/s/HEKBXwse9iSJwEZ) and extract the zip file to the main directory.

4. The training pipeline is outlined in the following [Jupyter Notebook](https://github.com/Houman-HM/VACNA/blob/main/Training%20VacnaNet.ipynb).
