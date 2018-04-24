#!/bin/bash
# Basic while loop
#python PNGFiles/resamplePNGFiles\ \(copy\).py 
## declare an array variable
declare -a arr=("PlasticCup" "Duck" "GolfBall" "Stapler")
mkdir train/
mkdir validate/
## now loop through the above array
for i in "${arr[@]}"
do
	mkdir train/$i/
	mkdir validate/$i/ 
	for f in PNGFiles/*$i*.jpeg; do mv $f train/$i/.; done
	for f in  train/$i/*${i}3*; do mv $f  validate/$i/.; done
	for f in  train/$i/*${i}7*; do mv $f  validate/$i/.; done
	#for f in  *.jpeg;  do rm $f ; done
   # or do whatever with individual element of the array
done

# You can access them using echo "${arr[0]}", "${arr[1]}" also








#python build_image_data.py --train_directory=./train --output_directory=./  --validation_directory=./validate --labels_file=mylabels.txt   --train_shards=1 --validation_shards=2 --num_threads=1 

#python ../logregAnd2Layer.py 



