import os,sys
import cv2
import tensorflow as tf
import tensorflow_vgg.vgg16 as vgg
from net.utils import *
from net.net import *


class Model():
    
    def __init__(self, sess, lambda_s, lambda_f, lambda_tv, learning_rate):
        self.sess = sess
        self.lambda_s = lambda_s
        self.lambda_f = lambda_f
        self.lambda_tv = lambda_tv
        self.learning_rate = learning_rate
        self.vgg_o = vgg.Vgg16()
        self.vgg_s = vgg.Vgg16()
        self.vgg_c = vgg.Vgg16()
        self.__build_net__()
        
    def __build_net__(self):
        
        self.Style = tf.placeholder(tf.float32, [None, 224,224,3])
        self.Content = tf.placeholder(tf.float32, [None, 224,224,3])
        self.Output = Network(self.Content,[i for i in range(0,10)], 10)
        
        o_features = self.__Output_Loss__()
        
        self.loss_s = self.__Style_Loss__(o_features)
        self.loss_c = self.__Content_Loss__(o_features)
        loss_tv = self.lambda_tv * total_variation_loss(self.Output)
        
        

        
        self.Loss = self.loss_s + self.loss_c + loss_tv
        self.Optim = tf.train.AdamOptimizer(self.learning_rate).minimize(self.Loss)
        
    def __Output_Loss__(self):
        with tf.variable_scope('out_vgg'):
            self.vgg_o.build(self.Output/255.0)
        return [self.vgg_o.conv1_2, self.vgg_o.conv2_2, self.vgg_o.conv3_3, self.vgg_o.conv4_3]
    
    def __Style_Loss__(self, o_features):
        with tf.variable_scope('style_vgg'):
            self.vgg_s.build(self.Style/255.0)
        s_features = [self.vgg_s.conv1_2, self.vgg_s.conv2_2, self.vgg_s.conv3_3, self.vgg_s.conv4_3]
        gram_ = [gram_matrix(l) for l in s_features]
        gram = [gram_matrix(l) for l in o_features]
        loss_s = tf.zeros(1, tf.float32)
        
        for g, g_ in zip(gram, gram_):
            loss_s +=  tf.reduce_mean(self.lambda_s * tf.reduce_mean(tf.square(tf.subtract(g, g_)), [1, 2]))
            
        return loss_s
        
    def __Content_Loss__(self, o_features):
        with tf.variable_scope('content_vgg'):
            self.vgg_c.build(self.Content/255.0)
        c_features = [self.vgg_c.conv3_3]
        
        loss_f = tf.zeros(1, tf.float32)
        for f, f_ in zip(c_features, o_features[2:3]):
            loss_f +=  tf.reduce_mean(self.lambda_f * tf.reduce_mean(tf.square(tf.subtract(f, f_)), [1, 2, 3]))
            
        return loss_f
    
    def train(self, style, content):
        return self.sess.run([self.loss_c, self.loss_s,self.Loss, self.Optim], feed_dict = {self.Style: style,
                                                                   self.Content:content})
    def pred(self, style, content):
        return self.sess.run(self.Output, feed_dict = {self.Style: style,
                                                         self.Content:content})
