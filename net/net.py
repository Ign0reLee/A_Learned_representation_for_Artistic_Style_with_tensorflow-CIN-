import os,sys
import tensorflow as tf
from net.utils import *


def Network(X, labels, num_categories):

    kernels = [9, 3, 3, 3, 3, 3, 3, 3, 3, 3, 9]
    strides = [1, 2, 2, 1, 1, 1, 1, 1, 2, 2, 1]
    channels= [32, 64, 128, 128, 128, 128, 128, 128, 64, 32, 3]

    h = X

    for i, kernel in enumerate(kernels):

        if i<3:
            h = ReLU(conv2d(h, channels[i], kernel, strides[i], Padding='VALID', Name='encode_conv_' + str(i)))
            h = CIN(h, labels, num_categories,name='encode_CIN_' + str(i))
            print(h)

        elif i<8:
    
            h =  Residual_block(h, channels[i], kernel,strides[i], labels, num_categories, Name='Residual_block'+str(i-3))
            print(h)

        elif i<10:
            h = ReLU(upconv2d(h, channels[i], kernel, strides[i], Padding='VALID', Name='decode_conv_' + str(i-8)))
            h = CIN(h, labels, num_categories, name='decode_CIN_' + str(i))
            print(h)

        else:

            h = Sigmoid(conv2d(h, channels[i], kernel, strides[i], Padding='VALID', Name='decode_conv_' + str(i-8)))
            h = CIN(h, labels, num_categories, name='decode_CIN_' + str(i))
            
            print(h)


    return  tf.div(tf.subtract(h,tf.reduce_min(h)),tf.subtract(tf.reduce_max(h),tf.reduce_min(h))) * 255.
