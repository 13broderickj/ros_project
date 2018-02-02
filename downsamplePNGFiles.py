# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 18:43:24 2017

@author: jasebroderick
"""
import Image 
import os,time
dir_path = os.path.dirname(os.path.realpath(__file__))
for filename in os.listdir(dir_path):
    debris = filename.split('.')
    if( not debris[-1]=='JPEG'):continue
    im = Image.open(filename) # might be png, gif etc, for instance test1.png
    maxsize = (60, 38)
    im.thumbnail(maxsize)
    im.save(filename, 'JPEG')
    time.sleep( .5 )