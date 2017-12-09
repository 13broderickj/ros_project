#!/usr/bin/env python3
'''
Zhiang Chen/Joseph Broderick
July,2016
'''
"Save the list of names in directory, and converts them into png files. "

import os
import subprocess
wd = os.getcwd()
print("Current directory is \""+wd+"\"")


files = os.listdir(wd)
images = list()
for name in files:
	debris = name.split('.')
	if (debris[-1] == 'pcd')&(not debris[0].startswith('box')):
		images.append(name)
		#./pcl_pcd2png GolfBall1.pcd GolfBall1.png --scale auto --color mono --field z
		newName=name.replace('.pcd','.png')
		subprocess.call('./pcl_pcd2png '+name+' '+newName+' --scale auto --color mono --field z',shell=True)

file_name = wd+"/name_lists"
fo = open(file_name,'w')
file = ' '.join(images)
fo.write(file)
fo.close()
print("Saved!")
print("The number of images is "+str(len(images)))
print("The size of the saved file is %0.2fkB" % float(os.path.getsize(file_name)/1024))
