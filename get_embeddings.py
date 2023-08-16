import torch
import cv2
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.modeling import build_model
from detectron2.utils.visualizer import Visualizer
from detectron2 import model_zoo

# Load a pretrained Faster R-CNN model from Detectron2 model zoo
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))  # Use the appropriate config file
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set a threshold for post-processing
model = DefaultPredictor(cfg)
#model.eval()

# Load and preprocess your input image
input_image = cv2.imread("/home/mila/v/vaibhav.jade/scratch/intern/stud/datasets/coco2017/val2017/000000000139.jpg")  # Load your image here
#input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)  # Convert to RGB
#input_tensor = torch.as_tensor(input_image.astype("float32").transpose(2, 0, 1))

# Run inference
with torch.no_grad():
    predictions = model(input_image)

# Extract features from the feature maps
feature_maps = predictions[0]['instances'].get_fields()["roi_features"]  # Extract feature maps for ROIs
print(feature_maps)