from __future__ import print_function
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from scipy.spatial.distance import pdist, cdist, squareform

import torch
import cv2
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.modeling import build_model
from detectron2.utils.visualizer import Visualizer
from detectron2 import model_zoo

"""Train/eval script."""
import logging
import os
import os.path as osp
import time
from collections import OrderedDict
import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog
from detectron2.engine import default_setup, hooks, launch
from detectron2.evaluation import (
    COCOEvaluator,
    DatasetEvaluators,
    verify_results,
)

class Trainer(DefaultTrainer):
    
    def run_step(self):
        start = time.perf_counter()
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        def hook(module, input, output):
                features.append(output)
                
            
        h = self.model.backbone.register_forward_hook(hook)
        features = []
        predictions = self.model(data)[0]
        h.remove()
        # Get detection instances
        instances = predictions['instances']

        # Get detected boxes and extract corresponding feature embeddings
        pred_boxes = instances.pred_boxes  # Prediction boxes (x1, y1, x2, y2)
        feature_extractor = self.model.roi_heads.box_pooler  # ROI feature extractor

        # Convert prediction boxes to proper format for ROI pooling
        #num_boxes = pred_boxes.shape[0]
        #box_list = [pred_boxes[i].unsqueeze(dim=0) for i in range(num_boxes)]  # Assign batch index 0 to all boxes

        # Extract feature embeddings from the feature map for each detected box
        with torch.no_grad():
            roi_features = feature_extractor([features[0][f] for f in self.model.roi_heads.in_features], [pred_boxes])

        return predictions, roi_features
