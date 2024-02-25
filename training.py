import torch
import cv2
import os
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.modeling import build_model
from detectron2.utils.visualizer import Visualizer
from detectron2 import model_zoo
from detectron2.data import build_detection_train_loader,get_detection_dataset_dicts,build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
import pickle
import time
#import wandb

#register dataset
def register_esmart_wip(root = "stud/datasets/esmart/"):
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
                        os.path.join(root, 'labels.json'),
                        os.path.join(root, 'data/'),
                    )

class Trainer(DefaultTrainer):

    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        """
        super().__init__(cfg)
        for param in self.model.backbone.parameters():
             param.requires_grad = False
        for param in self.model.backbone.parameters():
             param.requires_grad = False
        

register_esmart_wip()
cfg = get_cfg()
#cfg.merge_from_file("/home/vaibhav/Desktop/stud/configs/BDD100k/stud_resnet.yaml")
#cfg = model_zoo.get_config(config_path="COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",trained=True)  # Use the appropriate config file
cfg.merge_from_file("ood_distance/finetune_bigdet_trained.yaml")
# wandb.init(project="finetune-coco-run-esmart", name = cfg.OUTPUT_DIR.split('/')[-1],
#                #config = args.config_file
#                )
#cfg.MODEL.WEIGHTS = "/home/vaibhav/Desktop/stud/models/model_final_resnet_bdd.pth"
trainer = Trainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
