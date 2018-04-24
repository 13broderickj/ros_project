# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 18:43:24 2017

@author: jasebroderick
"""
import os,time
from PIL import Image, ImageChops,ImageMath,ImageOps

def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)
def bootstrapSample(im,shift):
    imCopyLeft=im.copy()
    imCopyRight=im.copy()
    imCopyUp=im.copy()
    imCopyDown=im.copy()
    xLen,yLen=im.size
    #
    for x in range(xLen):
           for y in range(yLen):
               imCopyLeft.putpixel((x, y), im.getpixel(((x+shift)%xLen,y)))
    for x in range(xLen):
           for y in range(yLen):
               imCopyRight.putpixel((x, y), im.getpixel(((x-shift)%xLen,y)))
    for x in range(xLen):
           for y in range(yLen):
               imCopyUp.putpixel((x, y), im.getpixel((x,(y+shift)%yLen)))
    for x in range(xLen):
           for y in range(yLen):
               imCopyDown.putpixel((x, y), im.getpixel((x,(y-shift)%yLen)))
    return (imCopyLeft,imCopyRight,imCopyUp,imCopyDown) 



dir_path = os.path.dirname(os.path.realpath(__file__))
for filename in os.listdir(dir_path):
    debris = filename.split('.')
    if( not debris[-1]=='png'):continue
    im = Image.open(filename) # might be png, gif etc, for instance test1.png
    imageCrop=im.crop((100,0,550,280))
    maxsize = (40, 40)
    imageCrop=ImageOps.fit(imageCrop, maxsize, Image.ANTIALIAS)
    im2 = ImageMath.eval('im/256', {'im':imageCrop}).convert('L')
    im2.save(filename.replace('png','jpeg'), 'JPEG')
    time.sleep(0.1)
    '''shifts=[i for i in range(15)]
    for shift in shifts:
        a,b,c,d=bootstrapSample(im2,shift)
        for imToModify in [a,b,c,d]:
            imageCrop=imToModify.crop((100,0,550,280))
            maxsize = (60, 36)
            imageCrop=ImageOps.fit(imageCrop, maxsize, Image.ANTIALIAS)
            im2 = ImageMath.eval('im/256', {'im':imageCrop}).convert('L')
            im2.save(str(shift)+filename.replace('png','jpeg'), 'JPEG')'''

