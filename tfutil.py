
# coding: utf-8

# In[2]:

import tensorflow as tf
import numpy as np
import math
import cv2
from matplotlib import pyplot as plt
import os
import time
import scipy.io


def adaptive_global_average_pool_2d(x):
    """
    In the paper, using gap which output size is 1, so i just gap func :)
    :param x: 4d-tensor, (batch_size, height, width, channel)
    :return: 4d-tensor, (batch_size, 1, 1, channel)
    """
    c = x.get_shape()[-1]
    return tf.reshape(tf.reduce_mean(x, axis=[1, 2]), (-1, 1, 1, c))

def prelu(_x):
    
    alphas = tf.get_variable('alpha', _x.get_shape()[-1], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
    pos = tf.nn.relu(_x)
    neg = alphas * (_x - abs(_x)) * 0.5

    return pos + neg



def up_scaling(x,f,scale_factor,name):
    with tf.variable_scope(name):
        
        if scale_factor == 3:
            x = tf.layers.conv2d(x, f * 9, 1, activation = None, name='conv2d-image_scaling-0')
            x = pixel_shuffle(x, 3)
        
        elif scale_factor & (scale_factor - 1) == 0:  # is it 2^n?
            log_scale_factor = int(np.log2(scale_factor))
            for i in range(log_scale_factor):
                x = tf.layers.conv2d(x, f * 4, 1, activation = None, name='conv2d-image_scaling-%d' % i)
                x = pixel_shuffle(x, 2)
        else:
            raise NotImplementedError("[-] Not supported scaling factor (%d)" % scale_factor)
        return x



    
def pixel_shuffle(x, scaling_factor):
    # pixel_shuffle
    # (batch_size, h, w, c * r^2) to (batch_size, h * r, w * r, c)
    sf = scaling_factor

    _, h, w, c = x.get_shape()
    c //= sf ** 2

    x = tf.split(x, scaling_factor, axis=-1)
    x = tf.concat(x, 2)

    x = tf.reshape(x, (-1, h * scaling_factor, w * scaling_factor, c))
    return x




def _phase_shift(I, r):
	return tf.depth_to_space(I, r)

def PS(X, r, color=False):
	if color:
		Xc = tf.split(X, 3, 3)
		X = tf.concat([_phase_shift(x, r) for x in Xc],3)
	else:
		X = _phase_shift(X, r)
	return X

def upsample(x, scale=8, features=256, activation=tf.nn.relu):
	assert scale in [2, 3, 4, 8]
	x = tf.layers.conv2d(x, features, [3,3], activation=activation, padding='SAME')
	if scale == 2:
		ps_features = 3*(scale**2)
		x = tf.layers.conv2d(x, ps_features, [3,3], activation=activation, padding='SAME') #Increase channel depth
		#x = slim.conv2d_transpose(x,ps_features,6,stride=1,activation_fn=activation)
		x = PS(x, 2, color=True)
	elif scale == 3:
		ps_features  =3*(scale**2)
		x = tf.layers.conv2d(x, ps_features, [3,3], activation=activation, padding='SAME')
		#x = slim.conv2d_transpose(x,ps_features,9,stride=1,activation_fn=activation)
		x = PS(x, 3, color=True)
	elif scale == 4:
		ps_features = 3*(2**2)
		for i in range(2):
			x = tf.layers.conv2d(x, ps_features, [3,3], activation=activation, padding='SAME')
			#x = slim.conv2d_transpose(x,ps_features,6,stride=1,activation_fn=activation)
			x = PS(x, 2, color=True)
	elif scale == 8:
		ps_features = 3*(8*8)
		x = tf.layers.conv2d(x, ps_features, [3,3], activation=activation, padding='SAME')
		#x = tf.layers.conv2d_transpose(x,3,kernel_size=(3,3),strides=2,activation=activation, padding='SAME')
		x = PS(x, 8, color=True)
	return x





