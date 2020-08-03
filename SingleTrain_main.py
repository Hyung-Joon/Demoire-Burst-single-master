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
from os import path
from PIL import Image
from C_Unet import C_Unet
import Model
import tfutil as tfu

def random_crop(lr_img, hr_img, hr_crop_size):
    lr_crop_size = hr_crop_size

    lr_w = np.random.randint(lr_img.shape[1] - lr_crop_size + 1)
    lr_h = np.random.randint(lr_img.shape[0] - lr_crop_size + 1)

    hr_w = lr_w
    hr_h = lr_h

    lr_img_cropped = lr_img[lr_h:lr_h + lr_crop_size, lr_w:lr_w + lr_crop_size]
    hr_img_cropped = hr_img[hr_h:hr_h + hr_crop_size, hr_w:hr_w + hr_crop_size]

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

def scan_over_dirs(d):
    file_list = []
    for dir_name in os.listdir(d):
        dir_full = path.join(d, dir_name, '*.png')
        files = sorted(glob.glob(dir_full))

        file_list.extend(files)

    return file_list

def rgb2ycbcr(img):
    y = 16 + (65.738 * img[:, :, 0]) + (129.057 * img[:, :, 1]) + (25.064 * img[:, :, 2])
    return y / 256

def augment(img_x, img_y):
    rand = np.random.rand()
    if rand > .5:
        img_x = cv2.flip(img_x,0)
        img_y = cv2.flip(img_y,0)
    rand = np.random.rand()
    if rand > .5:
        img_x = cv2.flip(img_x,1)
        img_y = cv2.flip(img_y,1)

    return img_x, img_y

def rotate(img_x, img_y):
    random_angle = int(random.random() * 5) * 90
    img_x = img_x.rotate(random_angle)
    img_y = img_y.rotate(random_angle)

    return img_x, img_y

def getBatch(iter, num, bs, patch_size, HRImages, LRImages):
    #id = np.random.choice(range(datalen),bs)
    x   = np.zeros( (bs, patch_size,patch_size,3), dtype=np.float32)
    y   = np.zeros( (bs, patch_size,patch_size,3), dtype=np.float32)
    for i in range(bs):
        img_lr = LRImages[num[(iter-1)*bs+i]]
        img_hr = HRImages[num[(iter-1)*bs+i]]

        img_x, img_y = random_crop(img_lr, img_hr,patch_size)
        img_x, img_y = augment(img_x, img_y)

        x[i,:,:,:] = (img_x) / 255.0
        y[i,:,:,:] = (img_y) / 255.0
    return x, y




gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # 텐서플로가 첫 번째 GPU만 사용하도록 제한
  try:
    tf.config.experimental.set_visible_devices(gpus[2], 'GPU')
  except RuntimeError as e:
    # 프로그램 시작시에 접근 가능한 장치가 설정되어야만 합니다
    print(e)


bs = 4
patch_size = 128
# LRDir = "/home/user/depthMap/test/datasets/TrainSingle/input"
# HRDir = "/home/user/depthMap/test/datasets/TrainSingle/gt"
# VAL_HR_Dir = "/home/user/depthMap/test/datasets/TrainSingle/val_gt"
# VAL_LR_Dir = "/home/user/depthMap/test/datasets/TrainSingle/val_input"
CP_Dir = "/home/user/depthMap/test/Demoire/blur_cp"

split = 'train'
split_path = 'train_crop'
target_dir='/home/user/depthMap/test/ntire-2020-deblur-mobile/REDS'
t_path_blur = path.join(target_dir, split_path, split + '_blur')
t_scan_blur = sorted(glob.glob(t_path_blur + "/*.png"))
t_path_sharp = path.join(target_dir, split_path, split + '_sharp')
t_scan_sharp = sorted(glob.glob(t_path_sharp + "/*.png"))

v_path_blur = path.join(target_dir, 'val', 'val' + '_blur')
v_scan_blur = sorted(glob.glob(t_path_blur + "/*.png"))
v_path_sharp = path.join(target_dir, 'val', 'val' + '_sharp')
v_scan_sharp = sorted(glob.glob(t_path_sharp + "/*.png"))
scans = [(b, s) for b, s, in zip(t_scan_blur, t_scan_sharp)]
random.shuffle(scans)
scans = scans[0:16000]
LRPath = []
HRPath = []

for i in range(len(scans)):
    a , b = scans[i]
    LRPath.append(a)
    HRPath.append(b)


# LRPath = sorted(glob.glob(LRDir + "/*.png"))   # fix input imgs error
# HRPath = sorted(glob.glob(HRDir + "/*.png"))
# val_LRPath = sorted(glob.glob(VAL_LR_Dir + "/*.png"))   # fix input imgs error
# val_HRPath = sorted(glob.glob(VAL_HR_Dir + "/*.png"))

start = time.time()
LRImages = [cv2.imread(img_path) for img_path in LRPath]
HRImages = [cv2.imread(img_path) for img_path in HRPath]
print("%.4f sec took reading for training images"%(time.time()-start))

val_LRImages = [cv2.imread(img_path) for img_path in v_scan_blur]
val_HRImages = [cv2.imread(img_path) for img_path in v_scan_sharp]
print("%.4f sec took reading for validation images"%(time.time()-start))

datalen = len(LRPath)
datalen2 = len(HRPath)
print("{} has {} files".format(t_path_blur, datalen))
print("{} has {} files".format(t_path_sharp, datalen2))

datalen3 = len(v_scan_blur)
datalen4 = len(v_scan_sharp)
print("{} has {} files".format(v_path_blur, datalen3))
print("{} has {} files".format(v_path_sharp, datalen4))

global_step = tf.Variable(0, trainable=False)

learning_rate = 0.0001

lamda = 0.0001

start_time = time.time()

X = tf.placeholder(tf.float32, [None, None, None, 3]) # LR Patch

Y = tf.placeholder(tf.float32, [None, None, None, 3]) # HR Patch


l1_loss1 = tf.reduce_mean(tf.abs(D_Unet(X) - Y))

loss_opt = tf.train.AdamOptimizer(learning_rate).minimize(l1_loss1)

val_net1 = C_Unet(X, reuse=True)

saver = tf.train.Saver(max_to_keep=0)

# save_path = 'D:/ntire/cp/SRGAN_const_clip_0.01_epoch_28.094.ckpt'

ckpt = tf.train.get_checkpoint_state(CP_Dir)

with tf.Session() as sess:
    with tf.device('/job:localhost/replica:0/task:0/device:GPU:0'):

        if ckpt:  # is checkpoint exist
            #last_model = ckpt.model_checkpoint_path
            last_model = "/home/user/depthMap/test/Demoire/d_cp/D_const_clip_0.01_epoch_44.124.ckpt"
            # last_model = ckpt.all_model_checkpoint_paths[0]
            print("load " + last_model)
            saver.restore(sess, last_model)  # read variable data
            print("succeed restore model")
        else:
            print("initializing variables")
            init = tf.global_variables_initializer()
            sess.run(init)



        l1_loss_n = 0
        #softmax_loss_n = 0

        for iter in range(400):
            num = list(range(datalen))
            out = random.sample(num, len(num))

            for epoch in range(1, int(datalen/bs)+1):
                print("epoch {}/{}...".format(iter, 400),
                    "step {}/{}...".format(epoch, int(datalen/bs)),
                      "l1_loss : {:.4f}".format(l1_loss_n),
                      "--- %.9s seconds ---" % (time.time() - start_time))

                learning_rate = 0.0001

                input_x, output_y = getBatch(epoch, out, bs, patch_size, HRImages, LRImages)

                _, l1_loss_n = sess.run([loss_opt, l1_loss1], feed_dict={X: input_x, Y: output_y})


                if epoch % (datalen/bs) == 0:

                    print('Validation')
                    print("--- %.9s seconds ---" % (time.time() - start_time))

                    val_psnr = 0
                    learning_rate = 0

                    for a in range(datalen3):
                        val_gt = val_HRImages[a]
                        val_test = val_LRImages[a]

                        width = val_gt.shape[0]
                        height = val_gt.shape[1]

                        val_test = (val_test) / 255.0
                        val_feed = np.zeros((1,width,height,3))
                        result_image2 = np.zeros((width,height,3))

                        val_feed[0,:,:,:] = val_test[:,:,:]

                        test = sess.run([val_net1], feed_dict= {X:val_feed})
                        result = test[0]  # (1,W,H,3)
                        result_test = np.zeros((width, height, 3))
                        result_test[:, :, :] = result[0, :, :, :]
                        result_test = (result_test) * 255.0

                        result_image2[:, :, 0] = result_test[:, :, 2]
                        result_image2[:, :, 2] = result_test[:, :, 0]
                        result_image2[:, :, 1] = result_test[:, :, 1]


                        val_psnr = val_psnr + BB(result_test, val_gt)

                    val_psnr = val_psnr / datalen3
                    print("val_psnr : %.3f" % (val_psnr))

                    cv2.imwrite('/home/user/depthMap/test/Demoire/blur_sample/%.3f.png' % (val_psnr), result_image2)


                    if val_psnr > 36:
                        saver.save(sess, "/home/user/depthMap/test/Demoire/blur_cp/D_const_clip_0.01_epoch_%.3f.ckpt" % (val_psnr))