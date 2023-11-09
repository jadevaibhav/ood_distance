register_esmart_wip()
# Build COCO test data loader
test_data_loader = build_detection_train_loader(cfg)
'''
    dataset = get_detection_dataset_dicts("esmart_wip"),
    mapper=None,  # You can provide a custom data mapper here if needed
    num_workers=1,  # Number of worker threads for data loading
    total_batch_size=4 
)'''

fea_path = "/home/vaibhav/Desktop/stud/datasets/esmart/GTfeatures"

for batch in test_data_loader:
    # Process the batch here
    for im in batch:
        input = [im['image'],im['height'],im['width'],im['instances']]
        #{"image": im['image'], "height": im['height'], "width": im['width']}
        preds, roi_features = predictor(input)
        with open(os.path.join(fea_path,im)+".pkl",'wb') as handle:
            pickle.dump({'preds':preds,'features':roi_features},handle)


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


###### original inference loop code ######
# Load and preprocess your input image
#input_image = cv2.imread("/home/vaibhav/Desktop/stud/datasets/coco2017/val2017/000000000139.jpg")  # Load your image here
#input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)  # Convert to RGB
#input_tensor = torch.as_tensor(input_image.astype("float32").transpose(2, 0, 1))
path = "/home/vaibhav/Desktop/stud/datasets/esmart/data"
fea_path = "/home/vaibhav/Desktop/stud/datasets/esmart/features"

for im in os.listdir(path):
    input_image = cv2.imread(os.path.join(path,im))
    preds, roi_features = predictor(input_image)
    with open(os.path.join(fea_path,im)+".pkl",'wb') as handle:
        pickle.dump({'preds':preds,'features':roi_features},handle)
