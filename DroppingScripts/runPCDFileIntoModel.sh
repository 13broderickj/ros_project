#!/bin/bash
# Basic while loop
rosrun example_tf_listener kinect_to_world pcdFileToUse.pcd
./pcl_pcd2png pcdFileToUse.pcd pcdFileToUse.png --scale auto --color mono --field z
python downsamplePNGFiles.py 
python feedFileIntoModel.py --picture_file pcdFileToUse.JPEG 
echo All done

