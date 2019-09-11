# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 21:22:54 2019

@author: Melon
"""

import tensorflow as tf
import os

import yolov3_tiny, utils

if __name__ == '__main__':        
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    sess = tf.Session()
    
    image_h, image_w = 480, 640
    CLASSES          = ['rubbish']
    anchors          = utils.get_anchors('./anchors.txt', image_h, image_w)
    num_classes      = len(CLASSES)
    
    ckpt_file = './checkpoint/yolov3_mix_v2_tiny_low_q/Rubbish.ckpt-20000'
    model = yolov3_tiny.yolov3_tiny(num_classes, anchors)

    with tf.Graph().as_default() as graph:
        sess = tf.Session(graph=graph)
        inputs = tf.placeholder(tf.float32, [1, image_h, image_w, 3]) # placeholder for detector inputs
        print("=>", inputs)

        with tf.variable_scope('yolov3-tiny'):
            feature_map = model.forward(inputs, is_training=False)

        boxes, confs, probs = model.predict(feature_map)
        scores = confs * probs
        print("=>", boxes.name[:-2], scores.name[:-2])
        cpu_out_node_names = [boxes.name[:-2], scores.name[:-2]]
        boxes, scores, labels = utils.gpu_nms(boxes, scores, num_classes,
                                              score_thresh=0.5,
                                              iou_thresh=0.5)
        print("=>", boxes.name[:-2], scores.name[:-2], labels.name[:-2])
        gpu_out_node_names = [boxes.name[:-2], scores.name[:-2], labels.name[:-2]]
        feature_map_1, feature_map_2, feature_map_3 = feature_map
        saver = tf.train.Saver(var_list=tf.global_variables(scope='yolov3-tiny'))
    
        saver.restore(sess, ckpt_file)
        print('=> checkpoint file restored from ', ckpt_file)
        utils.freeze_graph(sess, './pb/yolov3_cpu_nms_tiny_mix_v2_low_q_5k.pb', cpu_out_node_names)
        utils.freeze_graph(sess, './pb/yolov3_gpu_nms_tiny_mix_v2_low_q_5k.pb', gpu_out_node_names)
