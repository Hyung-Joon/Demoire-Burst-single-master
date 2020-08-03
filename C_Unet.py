# coding: utf-8

# In[ ]:

import tensorflow as tf
import numpy as np
import math
import cv2
from matplotlib import pyplot as plt
import os
import time
import scipy.io
import Model
import tfutil as tfu


def atrous_conv2d(value, filters, rate, padding, name=None):
    conv = tf.layers.conv2d(inputs=value, filters=filters, kernel_size=3, strides=1, padding='SAME', dilation_rate=rate,
                            name=name, activation=None)
    return conv


def channel_attention(x, outdim, kernel_size, strides, name):
    with tf.variable_scope("CA-%s" % name):
        skip_conn = x

        # skip_conn = tf.identity(x, name='identity')

        x = tfu.adaptive_global_average_pool_2d(x)

        x = tf.layers.conv2d(x, outdim // 16, kernel_size=kernel_size, strides=strides, padding='SAME', name="conv_1",
                             activation=None)
        x = tf.nn.leaky_relu(x, alpha=0.2)

        x = tf.layers.conv2d(x, outdim, kernel_size=kernel_size, strides=strides, padding='SAME', name="conv_2",
                             activation=None)
        x = tf.nn.sigmoid(x)

        x = tf.multiply(skip_conn, x)

        return x




def SCRAB(x, name):
    with tf.variable_scope("SCRAB-%s" % name):

        skip_conn2 = x

        x = tf.layers.conv2d(x, 128, kernel_size=3, padding='SAME', strides=1, activation=None)  # (B,2W,2H,64)
        x = tf.nn.leaky_relu(x, alpha=0.2)

        skip_conn = x

        CA = channel_attention(x, outdim=128, kernel_size=1, strides=1, name=name)

        CA = skip_conn + CA

        multi = tf.layers.conv2d(CA, 128, kernel_size=3, padding='SAME', strides=1, activation=None)  # (B,2W,2H,64)
        multi = tf.nn.leaky_relu(multi, alpha=0.2)

        multi = multi + skip_conn2

        return multi

def Pre_DB(x, name):
    with tf.variable_scope("Pre_DB%d"%name) as scope:
        skip_conn = x

        for i in range(3):
            x1 = Model.conv2d(x, 128, kernel_size=3, strides=1, padding='SAME', name="dense_%d"%i,
                         activation=tf.nn.leaky_relu)

            x = tf.concat([x1,x],3)

        x = Model.conv2d(x, 128, kernel_size=3, strides=1, padding='SAME', name="dense2",
                         activation=tf.nn.leaky_relu)
        x = x + skip_conn

        return x


def Post_DB(x, name):
    with tf.variable_scope("Post_DB%d" % name) as scope:
        skip_conn = x
        for i in range(3):
            x1 = Model.conv2d(x, 128, kernel_size=3, strides=1, padding='SAME', name="dense_%d" % i,
                              activation=tf.nn.leaky_relu)

            x = tf.concat([x1, x], 3)

        x = Model.conv2d(x, 128, kernel_size=1, strides=1, padding='SAME', name="dense2",
                         activation=tf.nn.leaky_relu)
        x = x + skip_conn

        return x



def Recon(x, name):
    with tf.variable_scope("Recon%d"%name) as scope:


        x1 = Model.conv2d(x, 256, kernel_size=3, strides=1, padding='SAME', name="recon_1",
                         activation=tf.nn.leaky_relu)

        skip_conn = x1

        x2 = Model.conv2d(x1, 256, kernel_size=3, strides=1, padding='SAME', name="recon_2",
                         activation=tf.nn.leaky_relu)

        x3 = Model.conv2d(x2, 256, kernel_size=3, strides=1, padding='SAME', name="recon_3",
                          activation=tf.nn.leaky_relu)
        x3 = x3 + skip_conn

        x4 = Model.conv2d(x3, 256, kernel_size=3, strides=1, padding='SAME', name="recon_4",
                          activation=tf.nn.leaky_relu)

        x5 = Model.conv2d(x4, 256, kernel_size=3, strides=1, padding='SAME', name="recon_5",
                          activation=tf.nn.leaky_relu)

        x5 = x3 + x5

        x6 = Model.conv2d(x5, 256, kernel_size=3, strides=1, padding='SAME', name="recon_6",
                          activation=tf.nn.leaky_relu)

        return x6

def C_Unet(x, reuse=False):
    with tf.variable_scope("D_Unet", reuse=reuse) as scope:
        if reuse: scope.reuse_variables()

        skip_conn = x

        x_d1 = Model.conv2d(x, 128, kernel_size=5, strides=2, padding='SAME', name="conv_d1",
                               activation=tf.nn.leaky_relu)

        x_d2 = Model.conv2d(x_d1, 128, kernel_size=5, strides=2, padding='SAME', name="conv_d2",
                            activation=tf.nn.leaky_relu)

        x_d3 = Model.conv2d(x_d2, 128, kernel_size=5, strides=2, padding='SAME', name="conv_d3",
                            activation=tf.nn.leaky_relu)

        x_d1 = Pre_DB(x_d1,1)
        x_d1 = SCRAB(x_d1,1)
        x_d1 = Post_DB(x_d1,1)
        #x_d1 = Recon(x_d1,1)
        x_d1= tf.layers.conv2d_transpose(x_d1, 128, kernel_size=9, padding='SAME', strides=2,
                                             activation=tf.nn.leaky_relu)

        x_d2 = Pre_DB(x_d2, 2)
        x_d2 = SCRAB(x_d2, 2)
        x_d2 = Post_DB(x_d2, 2)
        #x_d2 = Recon(x_d2, 2)
        x_d2 = tf.layers.conv2d_transpose(x_d2, 128, kernel_size=7, padding='SAME', strides=4,
                                          activation=tf.nn.leaky_relu)

        x_d3 = Pre_DB(x_d3, 3)
        x_d3 = SCRAB(x_d3, 3)
        x_d3 = Post_DB(x_d3, 3)
        #x_d3 = Recon(x_d3, 3)
        x_d3 = tf.layers.conv2d_transpose(x_d3, 128, kernel_size=5, padding='SAME', strides=8,
                                          activation=tf.nn.leaky_relu)

        x = Model.conv2d(x, 128, kernel_size=3, strides=1, padding='SAME', name="conv_d0",
                               activation=tf.nn.leaky_relu)

        x = Pre_DB(x, 4)
        x = SCRAB(x, 4)
        x = Post_DB(x, 4)
        #x = Recon(x, 4)

        concat_d1d2 = tf.concat([x_d1,x_d2],3)
        concat_d3d0 = tf.concat([x_d3,x],3)
        total = tf.concat([concat_d1d2,concat_d3d0],3)

        total = Model.conv2d(total, 128, kernel_size=3, strides=1, padding='SAME', name="total_conv",
                               activation=tf.nn.leaky_relu)

        total = Model.conv2d(total, 3, kernel_size=3, strides=1, padding='SAME', name="RGB",
                             activation=None)

        total = total + skip_conn

    return total

