import torch
import cv2
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.modeling import build_model
from detectron2.utils.visualizer import Visualizer
from detectron2 import model_zoo

class CustomPredictor(DefaultPredictor):

    def __call__(self, original_image):
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

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
            feature_extractor = self.model.roi_heads.box_pooler  # ROI feature extractor

            # Convert prediction boxes to proper format for ROI pooling
            #num_boxes = pred_boxes.shape[0]
            #box_list = [pred_boxes[i].unsqueeze(dim=0) for i in range(num_boxes)]  # Assign batch index 0 to all boxes

            # Extract feature embeddings from the feature map for each detected box
            with torch.no_grad():
                roi_features = feature_extractor([features[0][f] for f in self.model.roi_heads.in_features], [pred_boxes])

            return predictions, roi_features

# Load a pretrained Faster R-CNN model from Detectron2 model zoo
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))  # Use the appropriate config file
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set a threshold for post-processing
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
predictor = CustomPredictor(cfg)



# Load and preprocess your input image
input_image = cv2.imread("/home/vaibhav/Desktop/stud/datasets/coco2017/val2017/000000000139.jpg")  # Load your image here
#input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)  # Convert to RGB
#input_tensor = torch.as_tensor(input_image.astype("float32").transpose(2, 0, 1))

'''
model = build_model(cfg)
model.eval()

# Run inference
with torch.no_grad():
    predictions = model([input_tensor])

# Extract features from the feature maps
pred_boxes = predictions[0]['instances'].pred_boxes.tensor  # Extract feature maps for ROIs

feature_extractor = model.roi_heads.box_pooler  # ROI feature extractor

# Convert prediction boxes to proper format for ROI pooling
num_boxes = pred_boxes.shape[0]
box_list = [torch.tensor([0] * num_boxes), pred_boxes]  # Assign batch index 0 to all boxes

# Extract ROI feature maps based on the prediction boxes
roi_features = feature_extractor(
    [input_tensor], [box_list]
)

print("here....")
'''
preds, roi_features = predictor(input_image)
print("here....")