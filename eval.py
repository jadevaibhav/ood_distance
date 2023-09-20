import torch
import cv2
import os
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.modeling import build_model
#from detectron2.utils.visualizer import Visualizer
from detectron2 import model_zoo
from detectron2.data import build_detection_test_loader,get_detection_dataset_dicts
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data.datasets import register_coco_instances
import pickle

def register_esmart_wip(root = "/home/vaibhav/Desktop/stud/datasets/esmart/"):
        things_classes = [
                        "bicycle","bus","car","lane","lanes","motorcycle","person",
                        "roadwork_tcd","speed_limit","stop sign", "traffic light",
                        "truck"
                          ]
        name = 'esmart_wip'
        metadata = {"thing_classes":things_classes}
        register_coco_instances(
                        name,
                        metadata,
                        os.path.join(root, 'labels_mod.json'),
                        os.path.join(root, 'data/'),
                    )
        
register_esmart_wip()

cfg = model_zoo.get_config(config_path="COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",trained=True)  # Use the appropriate config file
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.DATASETS.TEST = ("esmart_wip",)
#cfg.MODEL.WEIGHTS = "/home/vaibhav/Desktop/stud/models/model_final_resnet_bdd.pth"

predictor = DefaultPredictor(cfg)


# Evaluate the model on the custom dataset
evaluator = COCOEvaluator("esmart_wip", cfg, distributed=False, output_dir="./output/directory")
val_loader = build_detection_test_loader(cfg, "esmart_wip")
inference_on_dataset(predictor.model, val_loader, evaluator)

# Print evaluation results
print(evaluator.evaluate())