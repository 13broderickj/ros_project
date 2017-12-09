#!/bin/bash
# Basic while loop
counter=1
name=basketball
roslaunch exmpl_models add_$name.launch height:=1.0
while [ $counter -le 2000 ]
do
z=0.$counter
echo $counter
echo $z
rosservice call /gazebo/set_model_state '{model_state: {model_name: '$name',pose: {position:{x: .830,z: 1.0 }} , twist: {angular:{z: '$z'}}}}'
rosrun example_tf_listener kinect_to_world $name$counter.pcd
((counter++))
done
rosservice call gazebo/delete_model $name
echo All done

