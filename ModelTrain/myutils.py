# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 21:48:03 2019

@author: Melon
"""

import numpy as np
from glob import glob
import multiprocessing.dummy as multiprocessing
import cv2
import random

def GetImg(dirList,ith):
    path = dirList[ith]
    img = cv2.imread(path)
    return img

def GetImgMultiprocess(dirList,threadNum=5):
    pool = multiprocessing.Pool(threadNum)
    imgTotals = []
    for ith in range(len(dirList)):
        imgTotal = pool.apply_async(GetImg, args=(dirList,ith))
        imgTotals.append(imgTotal)
    pool.close()
    pool.join()
    imgTotal = np.array([x.get() for x in imgTotals])
    return imgTotal

def gamma_trans(img,gamma):                                             
    #gamma建议范围0.5-1.5
    #具体做法先归一化到1，然后gamma作为指数值求出新的像素值再还原
    gamma_table = [np.power(x/255.0,gamma)*255.0 for x in range(256)]   
    #如果gamma>1, 新图像比原图像暗,如果gamma<1,新图像比原图像亮
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img,gamma_table)

def HSV_trans(img, h_change=1, s_change=1, v_change=1): ##hsv变换
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv =  np.float64(hsv)
    if h_change != 0: #random.random()
        k = random.random()*0.5 + 0.75
        b = random.random()*8 - 4 #
        hsv[...,0] = k*hsv[...,0] + b
        hsv[...,0][ hsv[...,0] <= 0] = 0
        hsv[...,0][ hsv[...,0] >= 180] = 180
    if  s_change != 0:
        k = random.random()*0.5 + 0.75
        b = random.random()*10 - 5
        hsv[...,1] = k*hsv[...,1] + b
        hsv[...,1][ hsv[...,1] <= 0] = 1
        hsv[...,1][ hsv[...,1] >= 255] = 255
    if  v_change != 0:
        k = random.random()*0.25 + 0.9
        b = random.random()*10 - 5
        hsv[...,2] = k*hsv[...,2] + b
        hsv[...,2][ hsv[...,2] <= 0] = 1
        hsv[...,2][ hsv[...,2] >= 255] = 255
    hsv = np.uint8(hsv)
    img_new = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img_new

def motion_blur(image, degree=12, angle=45):
    image = np.array(image)
    # 这里生成任意角度的运动模糊kernel的矩阵， degree越大，模糊程度越高
    M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(degree))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))
 
    motion_blur_kernel = motion_blur_kernel / degree
    blurred = cv2.filter2D(image, -1, motion_blur_kernel)
 
    # convert to uint8
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)
    return blurred

def EnhanceImg(img):
    if random.randint(0,1) == 1:
        img = HSV_trans(img.copy(), h_change=0, s_change=1, v_change=1)#BGR2HSV
    if random.randint(0,1) == 1:
        img = gamma_trans(img.copy(), random.random()*0.8 + 0.6)
    if random.randint(0,1) == 1:
        degree = random.randint(1,20)
        img = motion_blur(img,degree=degree)
    img = ((np.float32(img) / 255.) - 0.5)*2.
    return img

def EnhanceMultiprocess(imgs,threadNum=5):
    pool = multiprocessing.Pool(threadNum)
    imgTotals = []
    for img in imgs:
        imgTotal = pool.apply_async(EnhanceImg, args=(img,))
        imgTotals.append(imgTotal)
    pool.close()
    pool.join()
    imgTotal = np.array([x.get() for x in imgTotals])
    return imgTotal

class Gt_Creater(object):
    def __init__(self, image_h, image_w, anchors, num_classes):

        self.anchors     = anchors
        self.num_classes = num_classes
        self.image_h     = image_h
        self.image_w     = image_w
        
    def prepare_bbox(self, bbox_dir):
        y_true_list = []
        for _dir in glob(bbox_dir):
            with open(_dir,'r') as f:
                gt_boxes = []
                s = f.readlines()
                for i in s:
                    gt_box = np.array(i[:-1].split(' ')[1:]).astype(np.float32)
                    gt_box = gt_box*np.array([self.image_w,self.image_h,self.image_w,self.image_h])
                    gt_boxes.append(np.expand_dims(gt_box,0))
            gt_boxes = np.concatenate(gt_boxes,0)
            labels = np.array([[1] for i in range(gt_boxes.shape[0])])##只有垃圾一类
            gt_boxes = np.concatenate([gt_boxes,labels],-1)
            y_true_list.append(self.preprocess_true_boxes(gt_boxes))
        return y_true_list
        
    def preprocess_true_boxes(self, gt_boxes):
        """
        Preprocess true boxes to training input format
        Parameters:
        -----------
        :param true_boxes: numpy.ndarray of shape [T, 4]
                            T: the number of boxes in each image.
                            4: coordinate => x_min, y_min, x_max, y_max
        :param true_labels: class id
        :param input_shape: the shape of input image to the yolov3 network, [416, 416]
        :param anchors: array, shape=[9,2], 9: the number of anchors, 2: width, height
        :param num_classes: integer, for coco dataset, it is 80
        Returns:
        ----------
        y_true: list(3 array), shape like yolo_outputs, [13, 13, 3, 85]
                            13:cell szie, 3:number of anchors
                            85: box_centers, box_sizes, confidence, probability
        """
        num_layers = len(self.anchors) // 3
        anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]]
        grid_sizes = [[self.image_h//x, self.image_w//x] for x in (32, 16, 8)]
    
#        box_centers = (gt_boxes[:, 0:2] + gt_boxes[:, 2:4]) / 2 # the center of box
#        box_sizes =    gt_boxes[:, 2:4] - gt_boxes[:, 0:2] # the height and width of box
        box_centers = gt_boxes[:, 0:2]# the center of box
        box_sizes =    gt_boxes[:, 2:4] # the height and width of box
        
#        gt_boxes[:, 0:2] = box_centers
#        gt_boxes[:, 2:4] = box_sizes
    
        y_true_13 = np.zeros(shape=[grid_sizes[0][0], grid_sizes[0][1], 3, 5+self.num_classes], dtype=np.float32)
        y_true_26 = np.zeros(shape=[grid_sizes[1][0], grid_sizes[1][1], 3, 5+self.num_classes], dtype=np.float32)
        y_true_52 = np.zeros(shape=[grid_sizes[2][0], grid_sizes[2][1], 3, 5+self.num_classes], dtype=np.float32)
    
        y_true = [y_true_13, y_true_26, y_true_52]
        anchors_max =  self.anchors / 2.
        anchors_min = -anchors_max
        valid_mask = box_sizes[:, 0] > 0
    
        # Discard zero rows.
        wh = box_sizes[valid_mask]
        # set the center of all boxes as the origin of their coordinates
        # and correct their coordinates
        wh = np.expand_dims(wh, -2)
        boxes_max = wh / 2.
        boxes_min = -boxes_max
    
        intersect_mins = np.maximum(boxes_min, anchors_min)
        intersect_maxs = np.minimum(boxes_max, anchors_max)
        intersect_wh   = np.maximum(intersect_maxs - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area       = wh[..., 0] * wh[..., 1]
    
        anchor_area = self.anchors[:, 0] * self.anchors[:, 1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)
        # Find best anchor for each true box
        best_anchor = np.argmax(iou, axis=-1)
    
        for t, n in enumerate(best_anchor):
            for l in range(num_layers):
                if n not in anchor_mask[l]: continue
    
                i = np.floor(gt_boxes[t,0]/self.image_w*grid_sizes[l][1]).astype('int32')
                j = np.floor(gt_boxes[t,1]/self.image_h*grid_sizes[l][0]).astype('int32')
    
                k = anchor_mask[l].index(n)
                c = gt_boxes[t, 4].astype('int32')
                
                y_true[l][j, i, k, 0:4] = gt_boxes[t, 0:4]
                y_true[l][j, i, k,   4] = 1.
                y_true[l][j, i, k, 4+c] = 1.
        return y_true_13, y_true_26, y_true_52
