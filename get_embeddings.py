import torch
import cv2
import os
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.modeling import build_model
from detectron2.utils.visualizer import Visualizer
from detectron2 import model_zoo
from detectron2.data import build_detection_train_loader,get_detection_dataset_dicts
from detectron2.data.datasets import register_coco_instances
import pickle


# IMP: old version of detectron2 does not support batching in test loader, to run this use a seperate env with latest detectron2

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
        



class CustomPredictor(DefaultPredictor):

    def __call__(self, original_image):
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            [image,height,width,gt] = original_image

            inputs = {"image": image, "height": height, "width": width}
            
           
            def hook(module, input, output):
                features.append(output)
                
            
            h = self.model.backbone.register_forward_hook(hook)
            features = []
            predictions = self.model([inputs])[0]
            h.remove()
            # Get detection instances
            instances = predictions['instances']

            # Get detected boxes and extract corresponding feature embeddings
            pred_boxes = instances.pred_boxes  # Prediction boxes (x1, y1, x2, y2)
            roi_pooler = self.model.roi_heads.box_pooler  # ROI feature extractor
            feature_extractor = self.model.roi_heads.box_head

            # Convert prediction boxes to proper format for ROI pooling
            #num_boxes = pred_boxes.shape[0]
            #box_list = [pred_boxes[i].unsqueeze(dim=0) for i in range(num_boxes)]  # Assign batch index 0 to all boxes

            # Extract feature embeddings from the feature map for each detected box
            with torch.no_grad():
                pooled_features = roi_pooler([features[0][f] for f in self.model.roi_heads.in_features], [pred_boxes])
                roi_features = feature_extractor(pooled_features)

            return predictions['instances'].pred_classes.detach().cpu(), roi_features.detach().cpu()


class GTCustomPredictor(DefaultPredictor):

    def __call__(self, original_image):
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            [image,height,width,gt] = original_image

            inputs = {"image": image, "height": height, "width": width}
            
           
            def hook(module, input, output):
                features.append(output)
                
            
            h = self.model.backbone.register_forward_hook(hook)
            features = []
            predictions = self.model([inputs])[0]
            h.remove()
            # Get detection instances
            instances = gt

            # Get detected boxes and extract corresponding feature embeddings
            pred_boxes = instances.get_fields()['gt_boxes'].to('cuda')
            #instances.pred_boxes  # Prediction boxes (x1, y1, x2, y2)
            roi_pooler = self.model.roi_heads.box_pooler  # ROI feature extractor
            feature_extractor = self.model.roi_heads.box_head

            # Convert prediction boxes to proper format for ROI pooling
            #num_boxes = pred_boxes.shape[0]
            #box_list = [pred_boxes[i].unsqueeze(dim=0) for i in range(num_boxes)]  # Assign batch index 0 to all boxes

            # Extract feature embeddings from the feature map for each detected box
            with torch.no_grad():
                pooled_features = roi_pooler([features[0][f] for f in self.model.roi_heads.in_features], [pred_boxes])
                roi_features = feature_extractor(pooled_features)

            return instances.get_fields()['gt_classes'], roi_features.detach().cpu()
        


        
# Load a pretrained Faster R-CNN model from Detectron2 model zoo
cfg = get_cfg()
#cfg.merge_from_file("/home/vaibhav/Desktop/stud/configs/BDD100k/stud_resnet.yaml")

#cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))  # Use the appropriate config file
cfg.merge_from_file("ood_distance/finetune_bigdet_trained.yaml")

cfg.DATASETS.TRAIN = ("esmart_wip",)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 
#cfg.MODEL.ROI_HEADS.NUM_CLASSES = 600 # Set a threshold for post-processing
#cfg.MODEL.WEIGHTS = "/home/vaibhav/Desktop/stud/models/model_final_resnet_bdd.pth"
cfg.MODEL.WEIGHTS = "ood_distance/checkpoints/esmart/bigdet_finetune_on_esmart/model_final.pth"
#"/home/vaibhav/Desktop/ood_distance/checkpoints/esmart/coco_finetune_on_esmart/model_final.pth"
# #model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
predictor = CustomPredictor(cfg)


register_esmart_wip()
# Build COCO test data loader
test_data_loader = build_detection_train_loader(cfg)
'''
    dataset = get_detection_dataset_dicts("esmart_wip"),
    mapper=None,  # You can provide a custom data mapper here if needed
    num_workers=1,  # Number of worker threads for data loading
    total_batch_size=4 
)'''

fea_path = "ood_distance/checkpoints/esmart/bigdet_features/finetune/pooled"

for batch in test_data_loader:
    # Process the batch here
    for im in batch:
        input = [im['image'],im['height'],im['width'],im['instances']]
        #{"image": im['image'], "height": im['height'], "width": im['width']}
        preds, roi_features = predictor(input)
        with open(os.path.join(fea_path,im['file_name'].split('/')[-1])+".pkl",'wb') as handle:
            pickle.dump({'preds':preds,'features':roi_features},handle)

