#!/bin/bash
# Basic while loop
counter=1

while [ $counter -le 100 ]
do
z=1.0$counter
echo $counter
echo $z
rosservice call gazebo/delete_model banana
sleep 10
roslaunch exmpl_models add_banana.launch height:=$z
sleep 60
((counter++))
rosrun pcl_utils new_pcd_snapshot banana$counter.pcd
done

echo All done

