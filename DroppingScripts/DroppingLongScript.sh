#!/bin/bash
# Basic while loop
declare -a arr=("Stapler")

## now loop through the above array
for i in "${arr[@]}"
do
	counter=1
	name=$i
	roslaunch exmpl_models add_$name.launch height:=1.0
	while [ $counter -le 1000 ]
		do
		z=0.$counter
		echo $counter
		echo $z
		for j in {1..10}
			do 
			rosservice call /gazebo/set_model_state '{model_state: {model_name: '$name',pose: {position:{x: .830,z: 1.0 }} , twist: {angular:{x: '$z'}}}}'
			rosrun example_tf_listener kinect_to_world x_$j$name$counter.pcd kinect_link$j
			rosservice call /gazebo/set_model_state '{model_state: {model_name: '$name',pose: {position:{x: .830,z: 1.0 }} , twist: {angular:{y: '$z'}}}}'
			rosrun example_tf_listener kinect_to_world y_$j$name$counter.pcd kinect_link$j
			rosservice call /gazebo/set_model_state '{model_state: {model_name: '$name',pose: {position:{x: .830,z: 1.0 }} , twist: {angular:{z: '$z'}}}}'
			rosrun example_tf_listener kinect_to_world z_$j$name$counter.pcd kinect_link$j
			done 
		((counter++))
		done
	rosservice call gazebo/delete_model $name
done
echo All done
# You can access them using echo "${arr[0]}", "${arr[1]}" also

