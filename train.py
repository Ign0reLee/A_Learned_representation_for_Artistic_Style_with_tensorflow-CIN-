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

test_content_path ='./images/content/golden_gate.jpg'
test_content_img = image_resize([test_content_path])
test_style_img = image_resize(['./images/Monet/Water_Lilies.jpg'])





config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.allow_growth = True
#config.log_device_placement = True
config.allow_soft_placement = True


losses =[]

with tf.Session(config = config) as sess:
    model = Model(sess, lambda_s, lambda_f, lambda_tv, learning_rate)
    
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    sess.run(tf.initialize_all_variables())
    saver = tf.train.Saver()
    
    for epoch in range(n_epoch):
        batch = next_batch(style_path, content_path, batch_size)
        
        for i, b in enumerate(batch):
            st_time = time.time()
            
            st, ct = b
            loss_c, loss_s, loss, _ = model.train(st, ct)
            ed_time = time.time()
            losses.append(list(loss_c) + list(loss_s))
            print("Epoch : ", epoch, " Batch : ", i, ' content_loss : ', loss_c, ' style_loss : ', loss_s, " Loss : ", loss, " Time : ", ed_time - st_time)

            
            if i % 100 ==0:
                
                
                output = model.pred(test_style_img, test_content_img)
                r,g,b = cv2.split(output[0])
                output = cv2.merge([b,g,r])
                saver.save(sess, './Model/'+test_content_path.split('/')[-1]+'_'+str(i)+'.ckpt')
                cv2.imwrite('./result/'+test_content_path.split('/')[-1]+'_'+str(i)+'.jpg',output)
                
                
    output = model.pred(test_style_img, test_content_img)
    r,g,b = cv2.split(output[0])
    output = cv2.merge([b,g,r])
    saver.save(sess, './Model/'+test_content_path.split('/')[-1]+'_'+str(i)+'.ckpt')
    cv2.imwrite('./result/'+test_content_path.split('/')[-1]+'_'+str(i)+'.jpg',output)
    plt.figure(1)
    for i, loss in enumerate(losses):
        plt.plot(i, loss[0], linestyle='-', color='hotpink')
    plt.title("Content Loss")
    plt.xlabel("Train Step")
    plt.ylabel("Loss")
    plt.axis([0, 90000, 0, 1e10])
    plt.savefig("./Graph/Content_Loss.jpg", dpi=350)
    plt.figure(2)
    for i, loss in enumerate(losses):
        plt.plot(i, loss[1], linestyle='-', color='hotpink')
    plt.title("Style Loss")
    plt.xlabel("Train Step")
    plt.ylabel("Loss")
    plt.axis([0, 90000, 0, 1e10])
    plt.savefig("./Graph/Style_Loss.jpg", dpi=350)





