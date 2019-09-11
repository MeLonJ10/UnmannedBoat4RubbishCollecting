# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 19:42:27 2019

@author: Melon
"""

import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
from glob import glob
import os

import utils
from myutils import EnhanceImg

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    IMAGE_H, IMAGE_W = 640, 480
    classes = utils.read_coco_names('./classes.txt')
    num_classes = len(classes)
    
    cpu_nms_graph = tf.Graph()
#    input_tensor, output_tensors = utils.read_pb_return_tensors(cpu_nms_graph, "./pb/yolov3_cpu_nms_5k.pb",
#                                               ["Placeholder:0", "concat_9:0", "mul_6:0"])
    input_tensor, output_tensors = utils.read_pb_return_tensors(cpu_nms_graph, "./pb/yolov3_cpu_nms_tiny_mix_v2_low_q_5k.pb",
                                               ["Placeholder:0", "concat_9:0", "mul_6:0"])
    sess = tf.Session(graph=cpu_nms_graph)
    save_dir = './result/result_tiny_mix_v2_low_q_5k/'
    if not os.path.exists(save_dir): os.makedirs(save_dir)

    for i, image_path in enumerate(glob('./data/*.jpg')):
#        img = Image.open(image_path)
#        _img = (((np.array(img.copy())/255.)-0.5)*2.)[:,:,::-1]
        img = cv2.imread(image_path)
        img_enhance = EnhanceImg(img)
        img = Image.fromarray(img)
        boxes, scores = sess.run(output_tensors, feed_dict={input_tensor: np.expand_dims(img_enhance, axis=0)})
        boxes, scores, labels = utils.cpu_nms(boxes, scores, num_classes, score_thresh=0.5, iou_thresh=0.5)
        image = utils.draw_boxes(img, boxes, scores, labels, classes, [IMAGE_H, IMAGE_W], show=False)
        cv2.imwrite(save_dir+image_path.split('\\')[-1],np.array(image))#[:,:,::-1])
