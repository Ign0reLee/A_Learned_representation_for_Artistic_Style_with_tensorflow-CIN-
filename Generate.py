import os,sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_vgg.vgg16 as vgg
import time
from net.utils import *
from net.model import *
from net.net import *
from net.batch import *

n_epoch = 1
batch_size = 10
lambda_s = 1e1
lambda_f =1e0
lambda_tv = 1e-2
learning_rate = 1e-3


content_path = Path_Load('../Fast_Nueral_Style/train', batch_size)
style_path = './images/Monet/'
style_path = [os.path.join(style_path, x)  for x in os.listdir(style_path) if not os.path.isdir(x)]

test_content_path ='./images/content/hoovertowernight.jpg'
test_content_img = image_resize(list(np.tile([test_content_path], 10)))
test_style_img = image_resize(['./images/Monet/Water_Lilies.jpg'])





config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.allow_growth = True
#config.log_device_placement = True
config.allow_soft_placement = True


losses =[]

with tf.Session(config = config) as sess:
    model = Model(sess, lambda_s, lambda_f, lambda_tv, learning_rate)
    
   
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    sess.run(tf.initialize_all_variables())
    saver.restore(sess, './Model/golden_gate_8277.ckpt')
             
                
    output = model.pred(test_style_img, test_content_img)

    for i,o in enumerate(output):
        r,g,b = cv2.split(o)
        output = cv2.merge([b,g,r])
        cv2.imwrite('./result/'+test_content_path.split('/')[-1].split(".")[0] +'_'+ style_path[i].split('/')[-1].split(".")[0]+'_'+str(i)+'.jpg',output)
                
                
    

