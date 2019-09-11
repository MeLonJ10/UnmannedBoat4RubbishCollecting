# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 12:21:50 2019

@author: A-WIN10
"""

from glob import glob
import numpy as np

bbox_dir = './labels_kunming/*.txt'

for _dir in glob(bbox_dir):
    with open(_dir,'r') as f:
        try:
            gt_boxes = []
            s = f.readlines()
            for i in s:
                gt_box = np.array(i[:-1].split(' ')[1:]).astype(np.float32)
                gt_boxes.append(gt_box)
            if len(gt_boxes)==0:
                print(_dir)
        except:
            print(_dir)