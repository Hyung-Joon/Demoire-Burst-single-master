import tensorflow as tf
import numpy as np
import math
import cv2
from matplotlib import pyplot as plt
import os
import time
import scipy.io
import h5py
import random
import glob
from PIL import Image
from Burst_Unet import Burst_Unet
import Model
import tfutil as tfu



def random_crop(lr_img, hr_img):

    lr_img_cropped = lr_img[16:112, 16:112]
    hr_img_cropped = hr_img[16:112, 16:112]

    return lr_img_cropped, hr_img_cropped

def read_image(path):
    image = np.array(Image.open(path)).astype('float32')
    return image

def BB(img1,img2):
    mse = np.mean((img1-img2)**2)
    if mse==0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX/math.sqrt(mse))

def rgb2ycbcr(img):
    y = 16 + (65.738 * img[:, :, 0]) + (129.057 * img[:, :, 1]) + (25.064 * img[:, :, 2])
    return y / 256

def augment(img_x, img_y, img_y1, img_y2, img_y3, img_y4, img_y5, img_y6):
    rand = np.random.rand()
    if rand > .5:
        img_x = cv2.flip(img_x,0)
        img_y = cv2.flip(img_y,0)
        img_y1 = cv2.flip(img_y1, 0)
        img_y2 = cv2.flip(img_y2, 0)
        img_y3 = cv2.flip(img_y3, 0)
        img_y4 = cv2.flip(img_y4, 0)
        img_y5 = cv2.flip(img_y5, 0)
        img_y6 = cv2.flip(img_y6, 0)

    rand = np.random.rand()
    if rand > .5:
        img_x = cv2.flip(img_x,1)
        img_y = cv2.flip(img_y,1)
        img_y1 = cv2.flip(img_y1, 1)
        img_y2 = cv2.flip(img_y2, 2)
        img_y3 = cv2.flip(img_y3, 3)
        img_y4 = cv2.flip(img_y4, 4)
        img_y5 = cv2.flip(img_y5, 5)
        img_y6 = cv2.flip(img_y6, 6)

    return img_x, img_y, img_y1, img_y2, img_y3, img_y4, img_y5, img_y6


def valBatch(j,bs, patch_size, HRImages, LRImages):
    #id = np.random.choice(range(datalen), bs)
    x = np.zeros((bs, patch_size, patch_size, 3), dtype=np.float32)
    y = np.zeros((bs, patch_size, patch_size, 18), dtype=np.float32)
    z = np.zeros((bs, patch_size, patch_size, 3), dtype=np.float32)

    img_lr = LRImages[7 * j + 3]

    img_lr1 = LRImages[7 * j]
    img_lr2 = LRImages[7 * j + 1]
    img_lr3 = LRImages[7 * j + 2]
    img_lr4 = LRImages[7 * j + 4]
    img_lr5 = LRImages[7 * j + 5]
    img_lr6 = LRImages[7 * j + 6]

    img_hr = HRImages[j]

    img_lr7 = np.concatenate((img_lr1, img_lr2, img_lr3, img_lr4, img_lr5, img_lr6), axis=2)

    x[0, :, :, :] = (img_hr) / 255.0
    y[0, :, :, :] = (img_lr7) / 255.0
    z[0, :, :, :] = (img_lr) / 255.0

    return x, y, z

def getBatch(bs, datalen, patch_size, HRImages, LRImages):
    id = np.random.choice(range(datalen),bs)
    x   = np.zeros( (bs, patch_size,patch_size,3), dtype=np.float32)
    y   = np.zeros( (bs, patch_size,patch_size,18), dtype=np.float32)
    z = np.zeros((bs, patch_size, patch_size, 3), dtype=np.float32)
    for i,j in enumerate(id):

        img_lr = LRImages[7*j+3]

        img_lr1 = LRImages[7 * j]
        img_lr2 = LRImages[7*j+1]
        img_lr3 = LRImages[7*j+2]
        img_lr4 = LRImages[7*j+4]
        img_lr5 = LRImages[7*j+5]
        img_lr6 = LRImages[7*j+6]

        


        img_hr = HRImages[j]

        img_x, img_y = random_crop(img_hr, img_lr)
        img_y1, img_y2 = random_crop(img_lr1, img_lr2)
        img_y3, img_y4 = random_crop(img_lr3, img_lr4)
        img_y5, img_y6 = random_crop(img_lr5, img_lr6)

        img_x, img_y, img_y1, img_y2, img_y3, img_y4, img_y5, img_y6 = augment(img_x, img_y, img_y1, img_y2, img_y3, img_y4, img_y5, img_y6)

        img_lr7 = np.concatenate((img_y1, img_y2, img_y3, img_y4, img_y5, img_y6), axis=2)
    
        
        x[i,:,:,:] = (img_x) / 255.0
        y[i,:,:,:] = (img_lr7) / 255.0
        z[i,:,:,:] = (img_y) / 255.0
        
    return x, y, z

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # 텐서플로가 첫 번째 GPU만 사용하도록 제한
  try:
    tf.config.experimental.set_visible_devices(gpus[2], 'GPU')
  except RuntimeError as e:
    # 프로그램 시작시에 접근 가능한 장치가 설정되어야만 합니다
    print(e)

bs = 16
patch_size = 96
LRDir = "/home/user/depthMap/test/datasets/train/input"
HRDir = "/home/user/depthMap/test/datasets/train/gt"
VAL_HR_Dir = "/home/user/depthMap/test/datasets/train/val_gt"
VAL_LR_Dir = "/home/user/depthMap/test/datasets/train/val_input"
CP_Dir = "/home/user/depthMap/test/Demoire/cp"

LRPath = sorted(glob.glob(LRDir + "/*.png"))   # fix input imgs error
HRPath = sorted(glob.glob(HRDir + "/*.png"))
val_LRPath = sorted(glob.glob(VAL_LR_Dir + "/*.png"))   # fix input imgs error
val_HRPath = sorted(glob.glob(VAL_HR_Dir + "/*.png"))

start = time.time()
LRImages = [cv2.imread(img_path) for img_path in LRPath]
HRImages = [cv2.imread(img_path) for img_path in HRPath]
print("%.4f sec took reading for training images"%(time.time()-start))

val_LRImages = [cv2.imread(img_path) for img_path in val_LRPath]
val_HRImages = [cv2.imread(img_path) for img_path in val_HRPath]
print("%.4f sec took reading for validation images"%(time.time()-start))

datalen = len(LRPath)
datalen2 = len(HRPath)
print("{} has {} files".format(LRDir, datalen))
print("{} has {} files".format(HRDir, datalen2))

datalen3 = len(val_LRPath)
datalen4 = len(val_HRPath)
print("{} has {} files".format(VAL_LR_Dir, datalen3))
print("{} has {} files".format(VAL_HR_Dir, datalen4))


global_step = tf.Variable(0, trainable=False)

learning_rate = 0.0001

lamda = 0.0001

start_time = time.time()

X = tf.placeholder(tf.float32, [None, None, None, 3]) # LR Patch

Y = tf.placeholder(tf.float32, [None, None, None, 3]) # HR Patch

Z = tf.placeholder(tf.float32, [None, None, None, 18])



l1_loss = tf.reduce_mean(tf.abs(D_Unet(X,Z) - Y))

loss_opt = tf.train.AdamOptimizer(learning_rate).minimize(l1_loss)

val_net = D_Unet(X,Z,reuse=True)

saver = tf.train.Saver(max_to_keep=0)

# save_path = 'D:/ntire/cp/SRGAN_const_clip_0.01_epoch_28.094.ckpt'

ckpt = tf.train.get_checkpoint_state(CP_Dir)

with tf.Session() as sess:
    with tf.device('/job:localhost/replica:0/task:0/device:XLA_GPU:2'):
        if ckpt:  # is checkpoint exist
            #last_model = ckpt.model_checkpoint_path
            last_model = "/home/user/depthMap/test/Demoire/cp/D_const_clip_0.01_epoch_44.027.ckpt"
            # last_model = ckpt.all_model_checkpoint_paths[0]
            print("load " + last_model)
            saver.restore(sess, last_model)  # read variable data
            print("succeed restore model")
        else:
            print("initializing variables")
            init = tf.global_variables_initializer()
            sess.run(init)



        l1_loss_n = 0

        for epoch in range(1, 200000):
            print("step {}/{}...".format(epoch, 200000),
                  "loss : {:.4f}".format(l1_loss_n),
                  "--- %.9s seconds ---" % (time.time() - start_time))

            learning_rate = 0.0001

            output_y, input_x, input_x1 = getBatch(bs,datalen2,patch_size,HRImages,LRImages)

            _, l1_loss_n = sess.run([loss_opt, l1_loss], feed_dict={X: input_x1, Y: output_y, Z: input_x})


            if epoch % 500 == 0:

                print('Validation')
                print("--- %.9s seconds ---" % (time.time() - start_time))

                val_psnr = 0
                learning_rate = 0

                for a in range(datalen4):

                    val_y, val_x, val_x1 = valBatch(a,1, 128, val_HRImages, val_LRImages)
                    width = val_x.shape[1]
                    height = val_x.shape[2]

                    val_psnr_in = np.zeros((width, height, 3))

                    val_psnr_in[:,:,:] = val_y[0,:,:,:]
                    result_image2 = np.zeros((width,height,3))


                    test = sess.run([val_net], feed_dict= {X:val_x1, Z:val_x})
                    result = test[0]  # (1,W,H,3)
                    result_test = np.zeros((width, height, 3))
                    result_test[:, :, :] = result[0, :, :, :]
                    result_test = (result_test) * 255.0
                    val_psnr_in = (val_psnr_in) * 255.0

                    result_image2[:, :, 0] = result_test[:, :, 2]
                    result_image2[:, :, 2] = result_test[:, :, 0]
                    result_image2[:, :, 1] = result_test[:, :, 1]

                    result_test = rgb2ycbcr(result_test)
                    val_psnr_in = rgb2ycbcr(val_psnr_in)

                    val_psnr = val_psnr + BB(result_test, val_psnr_in)

                val_psnr = val_psnr / datalen4
                print("val_psnr : %.3f" % (val_psnr))

                cv2.imwrite('/home/user/depthMap/test/Demoire/sample/%.3f.png' % (val_psnr), result_image2)


                if val_psnr > 35:
                    saver.save(sess, "/home/user/depthMap/test/Demoire/cp/D_const_clip_0.01_epoch_%.3f.ckpt" % (val_psnr))