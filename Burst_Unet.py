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


        x = tf.layers.conv2d(x, 256, kernel_size=3, padding='SAME', strides=1, activation=None)  # (B,2W,2H,64)
        x = tf.nn.leaky_relu(x, alpha=0.2)

        skip_conn = x

        CA = channel_attention(x, outdim=256, kernel_size=1, strides=1, name=name)

        multi = tf.layers.conv2d(CA, 256, kernel_size=3, padding='SAME', strides=1, activation=None)  # (B,2W,2H,64)
        multi = tf.nn.leaky_relu(multi, alpha=0.2)

        multi = multi + skip_conn

        return multi


def Burst_Unet(x, y, reuse=False):
    with tf.variable_scope("D_Unet", reuse=reuse) as scope:
        if reuse: scope.reuse_variables()

        skip_conn = x

        conv_y = Model.conv2d(y, 256, kernel_size=3, strides=1, padding='SAME', name="conv_y",
                              activation=tf.nn.leaky_relu)

        conv1_1 = Model.conv2d(x, 256, kernel_size=3, strides=1, padding='SAME', name="conv1_1",
                               activation=tf.nn.leaky_relu)
        conv1_2 = Model.conv2d(conv1_1, 256, kernel_size=3, strides=1, padding='SAME', name="conv1_2",
                               activation=tf.nn.leaky_relu)

        conv1_2 = tf.concat([conv1_2,conv_y],axis=3)

        conv1_2_CA = SCRAB(conv1_2, name="conv1_2_CA")

        d_conv1_2 = Model.conv2d(conv1_2_CA, 256, kernel_size=3, strides=2, padding='SAME', name="d_conv1_2",
                                 activation=tf.nn.leaky_relu)

        conv2_1 = Model.conv2d(conv1_2, 256, kernel_size=3, strides=2, padding='SAME', name="conv2_1",
                               activation=tf.nn.leaky_relu)  # down
        conv2_2 = Model.conv2d(conv2_1, 256, kernel_size=3, strides=1, padding='SAME', name="conv2_2",
                               activation=tf.nn.leaky_relu)

        concat2_2 = tf.identity(conv2_2)

        d_concat1 = tf.concat([d_conv1_2, concat2_2], axis=3)
        d_concat1 = SCRAB(d_concat1, name="d_concat1_CA")
        d_concat1_conv = Model.conv2d(d_concat1, 256, kernel_size=3, strides=2, padding='SAME', name="d_concat1_conv",
                                      activation=tf.nn.leaky_relu)

        conv3_1 = Model.conv2d(conv2_2, 256, kernel_size=3, strides=2, padding='SAME', name="conv3_1",
                               activation=tf.nn.leaky_relu)  # down
        conv3_2 = Model.conv2d(conv3_1, 256, kernel_size=3, strides=1, padding='SAME', name="conv3_2",
                               activation=tf.nn.leaky_relu)
        concat3_2 = tf.identity(conv3_2)

        d_concat2 = tf.concat([d_concat1_conv, concat3_2], axis=3)
        d_concat2 = SCRAB(d_concat2, name="d_concat2_CA")
        d_concat2_conv = Model.conv2d(d_concat2, 256, kernel_size=3, strides=2, padding='SAME', name="d_concat2_conv",
                                      activation=tf.nn.leaky_relu)

        conv4_1 = Model.conv2d(conv3_2, 256, kernel_size=3, strides=2, padding='SAME', name="conv4_1",
                               activation=tf.nn.leaky_relu)  # down
        d_concat3 = tf.concat([d_concat2_conv, conv4_1], axis=3)
        d_concat3 = SCRAB(d_concat3, name="d_concat3_CA")
        conv4_2 = Model.conv2d(d_concat3, 256, kernel_size=3, strides=1, padding='SAME', name="conv4_2",
                               activation=tf.nn.leaky_relu)
        # concat4_2 = tf.identity(conv4_2)

        u_conv4_2 = tf.identity(conv4_2)
        u_conv4_2 = tf.layers.conv2d_transpose(u_conv4_2, 256, kernel_size=3, padding='SAME', strides=8,
                                               activation=tf.nn.leaky_relu)

        ##################################################### Upsampling ##########################################################

        # conv7_1 = Model.upscale2d(conv6_2)
        conv7_1 = tf.layers.conv2d_transpose(conv4_2, 256, kernel_size=3, padding='SAME', strides=2,
                                             activation=tf.nn.leaky_relu)
        conv7_1 = tf.concat([conv7_1, d_concat1_conv], axis=3)
        conv7_2 = Model.conv2d(conv7_1, 256, kernel_size=3, strides=1, padding='SAME', name="conv7_2",
                               activation=tf.nn.leaky_relu)
        conv7_3 = Model.conv2d(conv7_2, 256, kernel_size=3, strides=1, padding='SAME', name="conv7_3",
                               activation=tf.nn.leaky_relu)

        u_conv7_3 = tf.identity(conv7_3)
        u_conv7_3 = tf.layers.conv2d_transpose(u_conv7_3, 256, kernel_size=3, padding='SAME', strides=4,
                                               activation=tf.nn.leaky_relu)

        u_concat1 = tf.concat([u_conv4_2, u_conv7_3], axis=3)

        # conv8_1 = Model.upscale2d(conv7_3)
        conv8_1 = tf.layers.conv2d_transpose(conv7_3, 256, kernel_size=3, padding='SAME', strides=2,
                                             activation=tf.nn.leaky_relu)
        conv8_1 = tf.concat([conv8_1, d_conv1_2], axis=3)
        conv8_2 = Model.conv2d(conv8_1, 256, kernel_size=3, strides=1, padding='SAME', name="conv8_2",
                               activation=tf.nn.leaky_relu)
        conv8_3 = Model.conv2d(conv8_2, 256, kernel_size=3, strides=1, padding='SAME', name="conv8_3",
                               activation=tf.nn.leaky_relu)

        u_conv8_3 = tf.identity(conv8_3)
        u_conv8_3 = tf.layers.conv2d_transpose(u_conv8_3, 256, kernel_size=3, padding='SAME', strides=2,
                                               activation=tf.nn.leaky_relu)

        u_concat2 = tf.concat([u_concat1, u_conv8_3], axis=3)

        # conv9_1 = Model.upscale2d(conv8_3)
        conv9_1 = tf.layers.conv2d_transpose(conv8_3, 256, kernel_size=3, padding='SAME', strides=2,
                                             activation=tf.nn.leaky_relu)
        conv9_1 = tf.concat([conv9_1, conv1_2_CA], axis=3)
        conv9_2 = Model.conv2d(conv9_1, 256, kernel_size=3, strides=1, padding='SAME', name="conv9_2",
                               activation=tf.nn.leaky_relu)
        u_concat3 = tf.concat([u_concat2, conv9_2], axis=3)
        output = Model.conv2d(u_concat3, 256, kernel_size=3, strides=1, padding='SAME', name="output_1",
                              activation=tf.nn.leaky_relu)
        output = Model.conv2d(output, 3, kernel_size=3, strides=1, padding='SAME', name="output",
                              activation=None)
        # conv9_3 = Model.conv2d(conv9_2, 64, kernel_size=3, strides=1, padding='SAME', name="conv9_3", activation = tf.nn.leaky_relu)

        output = output + skip_conn

    return output

