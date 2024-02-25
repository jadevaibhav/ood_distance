import pickle
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.covariance import EmpiricalCovariance
import torch
#from training import register_esmart_wip,get_cfg,build_detection_test_loader
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.modeling import build_model
from detectron2 import model_zoo
from detectron2.structures import Boxes

import copy

embed_classwise = {}
fea_path = "ood_distance/checkpoints/esmart/coco_features/finetune/pooled"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
for im in os.listdir(fea_path):
    x = {}
    with open(os.path.join(fea_path,im), 'rb') as f:
        x = pickle.load(f)
        
    idx = 0
    for i in x['preds']:
        if embed_classwise.get(i.item()) == None:
            embed_classwise[i.item()] = [x['features'][idx].detach().cpu().numpy()]
        else:
            embed_classwise[i.item()].append(x['features'][idx].detach().cpu().numpy())
        idx += 1

def sample_estimator(embed_classwise,num_classes):
    classwise_mean = np.zeros((num_classes,embed_classwise[0][0].shape[0]),dtype='float32')
    precision = []
    X = None
    group_lasso = EmpiricalCovariance(assume_centered=False)
    for c in embed_classwise.keys():
        f_x = np.array(embed_classwise[c])
        classwise_mean[c] = f_x.mean(axis=0)
        f_x = f_x - classwise_mean[c][np.newaxis,:]
        if c == list(embed_classwise.keys())[0]:
             X = f_x
        else:
            np.concatenate((X,f_x),axis=0)
        ### finding inverse of covariance
    group_lasso.fit(X)
    precision = group_lasso.precision_
    #precision = np.array(precision)
   

    return torch.from_numpy(classwise_mean).float().to(device),torch.from_numpy(precision).float().to(device)


def get_features(model, input, instances = None):
    
    def hook(module, input, output):
                features.append(output)
    
    ### hooks to get feature maps from backbone 
    h = model.backbone.register_forward_hook(hook)
    features = []
    predictions = model([input])[0]
    h.remove()

    # Get detection instances
    if instances == None:
        instances = predictions['instances']

    # Get detected boxes and extract corresponding feature embeddings
    ### detaching predictions from computation graph
    pred_boxes = Boxes(instances.pred_boxes.tensor.detach())  # Prediction boxes (x1, y1, x2, y2)
    roi_pooler = model.roi_heads.box_pooler  # ROI feature extractor
    feature_extractor = model.roi_heads.box_head

    pooled_features = roi_pooler([features[0][f] for f in model.roi_heads.in_features], [pred_boxes])
    roi_features = feature_extractor(pooled_features)

    return instances, roi_features

def eval_ood_on_data(model,test_loader,num_classes, sample_mean, precision,magnitude):
     ### a FasterRCNN model (we are not using trainer or predictor of detectron2)
    model.eval()
    model.requires_grad_(False)
    copy_model = copy.deepcopy(model)
    copy_model.eval()
    copy_model.requires_grad_(False)
    sample_mean.requires_grad_(False)
    precision.requires_grad_(False)

    Mahalanobis_score = []
    for batch in test_loader:
        for sample in batch:
            score = get_Mahalanobis_score(model,copy_model,sample,num_classes, sample_mean, precision,magnitude)
            Mahalanobis_score.append(score)

def get_Mahalanobis_score(model, copy_model, input, num_classes, sample_mean, precision,magnitude):
    '''
    Compute the proposed Mahalanobis confidence score on input dataset
    return: Mahalanobis score from layer_index
    '''
    
    Mahalanobis = []
    
    inp_image,h,w = input['image'],input['height'],input['width']
    ### the backward graph settings             
    inp_image = inp_image.to(device=device).to(dtype=torch.float32).requires_grad_(True)   

    im = {"image": inp_image, "height": h, "width": w}
    ### forward pass and roi pooling features
    ### Warning! instances here is still connected to graph,
    ### pass it under no_grad() context or be careful    
    instances,out_features = get_features(model,im)

    # compute Mahalanobis score
    gaussian_score = 0
    out_features = torch.unsqueeze(out_features,1)
    sample_mean = torch.unsqueeze(sample_mean,0)
    zero_f = out_features - sample_mean     #shape b,n_c,1024
    ### (b,n_c,1024) x (1024,1024) x (b,1024,n_c)= (b,n_c,n_c)
    gaussian_score = torch.matmul(torch.matmul(zero_f,precision), torch.transpose(zero_f,1,2))
    ### taking diag() of (n_c,n_c) matrices
    gaussian_score = gaussian_score*torch.eye(num_classes,num_classes,device=device)
    gaussian_score = -0.5*gaussian_score.sum(dim=-1)
    
    ### there's no need to calculate again as its the same calculation
    loss = -gaussian_score.max(dim=1)[0]
    
    for l in range(len(loss)):
        if l == len(loss) - 1:
            model.zero_grad()
            inp_image.grad = None
            loss[l].backward()
        else:
            model.zero_grad()
            inp_image.grad = None
            loss[l].backward(retain_graph=True)
        ### sign function on gradients
        gradient =  torch.ge(inp_image.grad, 0)
        gradient = (gradient.float() - 0.5) * 2
    
        ### scaling the gradient by image preprocessing std in original code, 
        ### not needed as its 1 in FasterRCNN, subject to specific case

        # gradient.index_copy_(1, torch.LongTensor([0]).cuda(), gradient.index_select(1, torch.LongTensor([0]).cuda()) / (model.pixel_std[0]))
        # gradient.index_copy_(1, torch.LongTensor([1]).cuda(), gradient.index_select(1, torch.LongTensor([1]).cuda()) / (model.pixel_std[1]))
        # gradient.index_copy_(1, torch.LongTensor([2]).cuda(), gradient.index_select(1, torch.LongTensor([2]).cuda()) / (model.pixel_std[2]))
    
        tempInputs = torch.add(inp_image, -magnitude, gradient)

        with torch.no_grad():
            
            _,noise_out_features = get_features(copy_model,{"image": tempInputs, "height": h, "width": w},instances)

            noise_gaussian_score = 0
            noise_out_features = torch.unsqueeze(noise_out_features[l],0)
            noise_zero_f = noise_out_features - sample_mean.squeeze()     #shape b,n_c,1024
            ### (b,n_c,1024) x (1024,1024) x (b,1024,n_c)= (b,n_c,n_c)
            noise_gaussian_score = -0.5*torch.mm(torch.mm(noise_zero_f,precision), noise_zero_f.t()).diag()
            
            noise_gaussian_score = torch.max(noise_gaussian_score)
            Mahalanobis.append(noise_gaussian_score.cpu().numpy())
        
    return [[instances.pred_boxes.tensor.detach().cpu().numpy(),instances.pred_classes.detach().cpu().numpy()],Mahalanobis]


register_esmart_wip()
cfg = get_cfg()
#cfg.merge_from_file("/home/vaibhav/Desktop/stud/configs/BDD100k/stud_resnet.yaml")
#cfg = model_zoo.get_config(config_path="COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",trained=False)  # Use the appropriate config file
cfg.merge_from_file("ood_distance/finetune_coco_trained.yaml")
#cfg._BASE_ = "/home/mila/v/vaibhav.jade/scratch/intern/stud/configs/Base-RCNN-FPN.yaml"
cfg.DATASETS.TEST = ("esmart_wip",)

cfg.MODEL.WEIGHTS = "ood_distance/checkpoints/esmart/coco_finetune_on_esmart/model_final.pth"
test_data_loader = build_detection_test_loader(cfg,"esmart_wip")

model = build_model(cfg)
sample_mean,precision = sample_estimator(embed_classwise,12)

eval_ood_on_data(model,test_data_loader,12,sample_mean,precision,0.1)
