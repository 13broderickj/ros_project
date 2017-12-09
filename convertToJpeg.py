# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 18:43:24 2017

@author: jasebroderick
"""
import Image 
import os,time
dir_path = os.path.dirname(os.path.realpath(__file__))
for filename in os.listdir(dir_path):
    if('convertToJpeg.py'==filename):continue
    im = Image.open(filename) # might be png, gif etc, for instance test1.png
    #im.thumbnail(size, Image.ANTIALIAS) # size is 640x480
    im.convert('RGB').save(filename.replace('bmp','jpeg'), 'JPEG')
    time.sleep( 5 )