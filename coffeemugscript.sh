#!/bin/bash
# Basic while loop
counter=1

while [ $counter -le 100 ]
do
z=1.0$counter
echo $counter
echo $z
rosservice call gazebo/delete_model CoffeeMug
sleep 10
roslaunch exmpl_models add_coffee_mug.launch height:=$z
sleep 30
((counter++))
rosrun pcl_utils new_pcd_snapshot coffee_mug$counter.pcd
sleep 30
rosrun pcl_utils new_pcd_snapshot coffee_mug$counter_2.pcd
done

echo All done

