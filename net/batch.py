import os,sys
import numpy as np
import cv2
import tensorflow as tf
import tensorflow_vgg.vgg19 as vgg

def Path_Load(Content_Path, batch_size):
    path = [os.path.join(Content_Path, x) for x in os.listdir(Content_Path)]

    batch_scale = len(path)//batch_size

    return path[:batch_scale * batch_size]

def image_resize(img_path):
    
    im = []
    for path in img_path:
        img = cv2.imread(path)
        b,g,r = cv2.split(img)
        im.append(cv2.resize(cv2.merge([r,g,b]),(224,224)))
    return im

def next_batch(sp, cp, batch_size):
    
    
    for a in range(0, len(cp), batch_size):

        yield image_resize(sp), image_resize(cp[a:a+batch_size])
