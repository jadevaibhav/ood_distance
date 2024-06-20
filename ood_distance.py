import pickle
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.covariance import EmpiricalCovariance
import torch
#from training import register_esmart_wip,get_cfg,build_detection_test_loader
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader,build_detection_train_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.modeling import build_model
from detectron2 import model_zoo
from detectron2.structures import Boxes,Instances
import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
import cv2
import copy
from detectron2.layers import batched_nms

embed_classwise = {}
fea_path = "ood_distance/checkpoints/esmart/coco_features/gt/box"
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


def get_features(model, input, instances = None,is_inference=False):
    
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
    if is_inference:
        pred_boxes = Boxes(torch.concat([instances,predictions['instances'].pred_boxes.tensor],dim=0))  # Prediction boxes (x1, y1, x2, y2)
    else:
        pred_boxes = Boxes(instances)
    roi_pooler = model.roi_heads.box_pooler  # ROI feature extractor
    feature_extractor = model.roi_heads.box_head

    pooled_features = roi_pooler([features[0][f] for f in model.roi_heads.in_features], [pred_boxes])
    roi_features = feature_extractor(pooled_features)

    return pred_boxes.tensor, roi_features,predictions


def get_features_for_proposals(model, input, instances = None):
    def hook1(module, input, output):
            proposals.append(output)

    def hook2(module, input, output):
            features.append(output)
    
    ### hooks to get feature maps from backbone 
    h1 = model.proposal_generator.register_forward_hook(hook1)
    h2 = model.roi_heads.box_head.register_forward_hook(hook2)
    #h3 = model.roi_heads.box_predictor.register_forward_hook(hook2)
    features = []
    proposals =[]
    predictions = model([input])[0]
    h1.remove()
    h2.remove()
    #h3.remove()

    return proposals[0][0][0],features[0]

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
    image = cv2.imread("/home/mila/v/vaibhav.jade/scratch/intern/ood_distance/notebooks/montreal-construction.jpeg")
    height, width = image.shape[:2]
    num_classes = 12
    magnitude = 0.01
    aug = T.ResizeShortestEdge(
        [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )
    im = aug.get_transform(image).apply_image(image)
    im = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
    im.to(device)

    inputs = {"image": im, "height": height, "width": width}
    score = get_Mahalanobis_score(model,copy_model,inputs,num_classes, sample_mean, precision,magnitude)
    for batch in test_loader:
        for sample in batch:
            score = get_Mahalanobis_score(model,copy_model,sample,num_classes, sample_mean, precision,magnitude)
            Mahalanobis_score.append(score)

def get_Mahalanobis_score(model, copy_model, inputs, num_classes, sample_mean, precision,magnitude):
    '''
    Compute the proposed Mahalanobis confidence score on input dataset
    return: Mahalanobis score from layer_index
    '''

    Mahalanobis = []

    inp_image,h,w= inputs['image'],inputs['height'],inputs['width']
    #gt = input['instances'].gt_boxes.tensor.to(device)
    #gt = input['']
    ### the backward graph settings             
    inp_image = inp_image.to(device=device).to(dtype=torch.float32).requires_grad_(True)   

    im = {"image": inp_image, "height": h, "width": w}
    ### forward pass and roi pooling features
    ### Warning! instances here is still connected to graph,
    ### pass it under no_grad() context or be careful    
    # instances,out_features = get_features_for_proposals(model,im)

    # out_nms = batched_nms(instances.proposal_boxes.tensor,instances.objectness_logits,torch.ones_like(instances.objectness_logits),0.5)
    # min_idx = min(20,len(out_nms))
    # prop_boxes = instances.proposal_boxes.tensor.detach()[out_nms[:min_idx]]
    # out_features = out_features[out_nms[:min_idx]]
    # compute Mahalanobis score

    coco_preds =  coco_model([im])[0]
    prop_boxes,out_features,predictions = get_features(model,im,coco_preds['instances'].pred_boxes.tensor.detach(),is_inference=True)
    #prop_boxes,out_features = get_features(model,im,gt)    

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
    loss = -(gaussian_score).max(dim=1)[0]

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

            _,noise_out_features,_ = get_features(copy_model,{"image": tempInputs, "height": h, "width": w},prop_boxes)

            noise_gaussian_score = 0
            noise_out_features = torch.unsqueeze(noise_out_features[l],0)
            noise_zero_f = noise_out_features - sample_mean.squeeze()     #shape b,n_c,1024
            ### (b,n_c,1024) x (1024,1024) x (b,1024,n_c)= (b,n_c,n_c)
            noise_gaussian_score = -0.5*torch.mm(torch.mm(noise_zero_f,precision), noise_zero_f.t()).diag()

            noise_gaussian_score = torch.max(noise_gaussian_score)
            Mahalanobis.append(noise_gaussian_score.cpu().numpy())

    ood_preds = torch.tensor(np.asarray(Mahalanobis)) < -10000.0
    ood_preds = ood_preds*1
    final_preds = batched_nms(prop_boxes,torch.tensor(np.asarray(Mahalanobis)).abs(),torch.ones_like(ood_preds),0.7)

    pred_instances = Instances((h,w))
    pred_instances.pred_boxes = Boxes(prop_boxes)[final_preds]
    pred_instances.pred_classes = (ood_preds + 1)[final_preds]
    pred_instances.mahal_scores = torch.tensor(np.asarray(Mahalanobis))[final_preds]

    return pred_instances,predictions['instances']

#register dataset
def register_coco_val(root = "/home/mila/v/vaibhav.jade/scratch/intern/stud/datasets/coco/"):
        things_classes = ["bicycle","bus","car","lane","lanes","motorcycle","person",
                        "roadwork_tcd","speed_limit","stop sign", "traffic light",
                        "truck"]
        name = 'coco_val'
        metadata = {"thing_classes":things_classes}
        register_coco_instances(
                        name,
                        metadata,
                        os.path.join(root, 'annotations/instances_val2017.json'),
                        os.path.join(root, 'val2017/'),
                    )
register_esmart_wip()
cfg = get_cfg()
#cfg.merge_from_file("/home/vaibhav/Desktop/stud/configs/BDD100k/stud_resnet.yaml")
#cfg = model_zoo.get_config(config_path="COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",trained=False)  # Use the appropriate config file
cfg.merge_from_file("ood_distance/configs/finetune_coco_trained.yaml")
#cfg._BASE_ = "/home/mila/v/vaibhav.jade/scratch/intern/stud/configs/Base-RCNN-FPN.yaml"
cfg.DATASETS.TEST = ("esmart_wip",)

cfg.MODEL.WEIGHTS = "ood_distance/checkpoints/esmart/coco_finetune_on_esmart/model_final.pth"
test_data_loader = build_detection_train_loader(cfg)

model = build_model(cfg)
checkpointer = DetectionCheckpointer(model)
checkpointer.load(cfg.MODEL.WEIGHTS)
sample_mean,precision = sample_estimator(embed_classwise,12)

cfg.merge_from_file("/home/mila/v/vaibhav.jade/scratch/intern/ood_distance/configs/coco_trained.yaml")
cfg.MODEL.WEIGHTS = "/home/mila/v/vaibhav.jade/scratch/intern/ood_distance/checkpoints/model_final_b275ba.pkl"

coco_model = build_model(cfg)
checkpointer = DetectionCheckpointer(coco_model)
checkpointer.load(cfg.MODEL.WEIGHTS)
coco_model.eval()

maha_score_id = eval_ood_on_data(model,test_data_loader,12,sample_mean,precision,0.1)

with open('maha_score_id.pickle', 'wb') as f:
    pickle.dump(maha_score_id, f)