# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 18:43:24 2017

@author: jasebroderick
"""
import os,time,random
import numpy as np
import re
from PIL import Image, ImageChops,ImageMath,ImageOps
from hyperopt import hp,fmin,tpe,STATUS_OK,rand,Trials
import hyperopt.pyll
from hyperopt.pyll import scope
import tensorflow as tf
import pickle
from threading import Thread
from time import sleep
def mutual_information(imageA,imageB):
     """ Mutual information for joint histogram
     """
     image1=np.array(imageA)
     image2=np.array(imageB)
     hist_2d, x_edges, y_edges = np.histogram2d(
     image1.ravel(),
     image2.ravel(),
     bins=20)
     # Convert bins counts to probability values
     pxy = hist_2d / float(np.sum(hist_2d))
     px = np.sum(pxy, axis=1) # marginal for x over y
     py = np.sum(pxy, axis=0) # marginal for y over x
     px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals
     # Now we can do the calculation using the pxy, px_py 2D arrays
     nzs = pxy > 0 # Only non-zero pxy values contribute to the sum
     return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))

def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)
def bootstrapSample(im,shift,a,b,c):
    imCopyLeft=im.copy()
    imCopyRight=im.copy()
    imCopyUp=im.copy()
    imCopyDown=im.copy()
    xLen,yLen=im.size
    imCopy=im.copy()
    if isinstance(a, tuple): a=a[0]
    if isinstance(b, tuple): b=b[0]
    if isinstance(c, tuple): c=c[0]

    for x in range(xLen):
        for y in range(yLen):
            if(im.getpixel((x,y))>=int(a)):
                imCopy.putpixel((x, y), im.getpixel((x,y))+int(b))
            else:
                imCopy.putpixel((x, y),int(c))
    im=imCopy
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

def createImages(a,b,c,d,chenPhoto,cls):
    info=[]
    dir_path = os.path.dirname(os.path.realpath(__file__))+'/PNGFiles'
    numSimToConsider=30
    simPhotosLookedAt=0
    if(cls=='tball'): cls='tennisball'
    if(cls=='gball'): cls='GolfBall'
    if isinstance(a, tuple): a=a[0]
    if isinstance(b, tuple): b=b[0]
    if isinstance(c, tuple): c=c[0]
    if isinstance(d, tuple): d=d[0]
    if(c-a<=40 or d-b<=40): return -100
    for filename in os.listdir(dir_path):
        if(simPhotosLookedAt>numSimToConsider): break
        debris = filename.split('.')
        if( not debris[-1]=='png'):continue
        if(not re.search(cls, filename, re.IGNORECASE)): continue
        im = Image.open(dir_path+'/'+filename) # might be png, gif etc, for instance test1.png
        imageCrop=im.crop((a,b,c,d))# old one im.crop((100,0,550,280))#im.crop((100,150,550,380)) # ' 2 3 4 7 8 9 10 - no 5 6 
        maxsize = (40, 40)
        imageCrop=ImageOps.fit(imageCrop, maxsize, Image.ANTIALIAS)
        im2 = ImageMath.eval('im/256', {'im':imageCrop}).convert('L')
        xLen,yLen=im2.size
        if(not(xLen==40 and yLen==40)): return -100
        #im2.save(filename.replace('png','jpeg'), 'JPEG')
        shifts=[i for i in range(10)]
        for shift in shifts:
            lIm,rIm,uIm,dIm=bootstrapSample(im2,shift,240,-65,12)
            for imToModify in [lIm,rIm,uIm,dIm]:
                imageNew=ImageOps.fit(imToModify, maxsize, Image.ANTIALIAS)
                info.append(mutual_information(imageNew,chenPhoto))
        simPhotosLookedAt+=1
    return max(info)
    

@scope.define
def hyperparameterObjective(cls,a=100,b=0,c=550,d=280):
    if isinstance(cls, float): return cls
    cls=cls[0]
    numChenToConsider=20
    info=[]
    dir_path = os.path.dirname(os.path.realpath(__file__))+'/'+str(cls)
    chenPhotosLookedAt=0
    for filename in os.listdir(dir_path):
        if(chenPhotosLookedAt>numChenToConsider): break
        debris = filename.split('.')
        if( not debris[-1]=='png'):continue
        print(dir_path+'/'+filename)
        chenPhoto = Image.open(dir_path+'/'+filename).convert('L') # might be png, gif etc, for instance test1.png
        val=createImages(a,b,c,d,chenPhoto,cls)
        if(val==-100):
            return 10.1
        else:
            chenPhotosLookedAt+=1
            info.append(val)
    return -1*sum(info)/len(info)




def thread_function(arg):
    y=['stapler','duck','cup','tball','gball'][arg]
    spaceChoice=scope.hyperparameterObjective([hp.choice('cls',[y])],
                                              [hp.uniform('a',100,550)], 
                                             [hp.uniform('b',0,280)],
                                             [hp.uniform('c',100,550)],
                                            [hp.uniform('d',0,280)] )
    trials = Trials()
    best=fmin(hyperparameterObjective,
              space=spaceChoice,
              algo=rand.suggest,
              max_evals=500,
              trials=trials)
    with open('trials'+y+'500.pkl', 'wb') as output:
        pickle.dump(trials,output,-1)
    with open('best'+y+'500.pkl', 'wb') as output:
        pickle.dump(best,output,-1)
    print(best)
if __name__ == "__main__":
    classes=['stapler','duck','cup','tball','gball']
    threads=[]
    for i in range(len(classes)):    
        threads.append(Thread(target = thread_function, args=(i,)))
        threads[-1].start()
    for thread in threads:
        thread.join()
    print ("thread finished...exiting")
