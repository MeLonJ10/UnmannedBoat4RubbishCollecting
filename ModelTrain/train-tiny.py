# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 21:31:14 2019

@author: Melon
"""

import tensorflow as tf
from glob import glob
import random
import os
import numpy as np

import yolov3_tiny, utils
from myutils import Gt_Creater, GetImgMultiprocess, EnhanceMultiprocess

if __name__ == '__main__':        
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    sess = tf.Session()
    
    IMAGE_H, IMAGE_W = 480, 640
    BATCH_SIZE       = 16
    STEPS            = 20000
    LR               = 0.001 # if Nan, set 0.0005, 0.0001
    DECAY_STEPS      = 100
    DECAY_RATE       = 0.9
    SHUFFLE_SIZE     = 200
    CLASSES          = ['rubbish']
    DATASET          = 'yolov3_mix_v2_tiny_low_q/Rubbish'
    ANCHORS          = utils.get_anchors('./anchors.txt', IMAGE_H, IMAGE_W)
    NUM_CLASSES      = len(CLASSES)
    SAVE_INTERNAL    = 100
    
    bbox_dir = '../dataset/mix_v2_bbox/*.txt'
    imgs_dir = '../dataset/mix_v2/*.jpg'
    
    is_training = tf.placeholder(tf.bool)

    model = yolov3_tiny.yolov3_tiny(NUM_CLASSES, ANCHORS)
    
    inputs = tf.placeholder(dtype=tf.float32,shape=(None,IMAGE_H,IMAGE_W,3),name='IMAGES')
    y_true = [tf.placeholder(dtype=tf.float32,shape=(None,15,20,3,6),name='y_true_layer1'),
              tf.placeholder(dtype=tf.float32,shape=(None,30,40,3,6),name='y_true_layer2'),
              tf.placeholder(dtype=tf.float32,shape=(None,60,80,3,6),name='y_true_layer3')]
    
    with tf.variable_scope('yolov3-tiny'):
        pred_feature_map = model.forward(inputs, is_training=is_training)
        loss             = model.compute_loss(pred_feature_map, y_true)
        y_pred           = model.predict(pred_feature_map)
    
    tf.summary.scalar("loss/coord_loss", loss[1])
    tf.summary.scalar("loss/sizes_loss", loss[2])
    tf.summary.scalar("loss/confs_loss", loss[3])
    tf.summary.scalar("loss/class_loss", loss[4])
    
    global_step = tf.Variable(0, trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])
    write_op = tf.summary.merge_all()
    writer_train = tf.summary.FileWriter("./logs/train")
    
    learning_rate = tf.train.exponential_decay(LR, global_step, decay_steps=DECAY_STEPS,
                                               decay_rate=DECAY_RATE, staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    
    # set dependencies for BN ops
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss[0], global_step=global_step)
    
    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    saver = tf.train.Saver(max_to_keep=200)
    saver.restore(sess,'checkpoint/yolov3_1300_tiny_low_q/Rubbish.ckpt-20001')
    
    gt_creater = Gt_Creater(IMAGE_H, IMAGE_W, ANCHORS, NUM_CLASSES)
    y_true_np = gt_creater.prepare_bbox(bbox_dir)
    images = GetImgMultiprocess(glob(imgs_dir))#!!! BGR

    index = [i for i in range(images.shape[0])]  
    for step in range(STEPS):
        #准备数据
        if step % (images.shape[0]//BATCH_SIZE) == 0:
            print("===> Epoch %d [Loading Data]"% (step//(images.shape[0]//BATCH_SIZE)))
            images_enhance = EnhanceMultiprocess(images)
        _index = random.sample(index,BATCH_SIZE)
        _images = images_enhance[_index]
        _y_true_ = [y_true_np[i] for i in _index]
        _y_true_1 = np.concatenate([np.expand_dims(_y_true_[i][0],0) for i in range(BATCH_SIZE)],0)
        _y_true_2 = np.concatenate([np.expand_dims(_y_true_[i][1],0) for i in range(BATCH_SIZE)],0)
        _y_true_3 = np.concatenate([np.expand_dims(_y_true_[i][2],0) for i in range(BATCH_SIZE)],0)
        _y_true_ = [_y_true_1,_y_true_2,_y_true_3]
        feed_dict = {is_training:True,inputs:_images,y_true[0]:_y_true_1,
                     y_true[1]:_y_true_2,y_true[2]:_y_true_3}
        run_items = sess.run([train_op, write_op, y_pred, y_true] + loss,feed_dict=feed_dict)
        writer_train.add_summary(run_items[1], global_step=step)
        writer_train.flush() # Flushes the event file to disk
    
        print("=> STEP %6d [TRAIN]:\tloss_xy:%7.4f \tloss_wh:%7.4f \tloss_conf:%7.4f \tloss_class:%7.4f"
            %(step+1, run_items[5], run_items[6], run_items[7], run_items[8]))
        if step % 1000 == 0:
            saver.save(sess, save_path="./checkpoint/%s.ckpt"%DATASET, global_step=step+1)
    saver.save(sess, save_path="./checkpoint/%s.ckpt"%DATASET, global_step=step+1)
    