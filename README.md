# ood_distance
All the utility code, config and notebooks are used for the unknown object identification using the OOD framework. The following section describes the individual code.

## Codebase
1. get_embedings.py: code to generate feature representations from pre-trained FRCNN model for given bboxes (can be ground truth or predictions themselves)
2. training.py: script to fine-tune the box-head by loading the particular pre-trained models
3. eval.py: Evaluation script for to generate the COCO eval metric (MAP score) for the finetuned models
4. ood_distance.py: main script for Mahalanobis distance based OOD metric generation
5. Notebooks:
   1. feature_viz_finetune-(dataset).ipynb: for experiments for e-smart dataset feature visualization and linear separability test
   2. ood_distance_viz.ipynb: visualizing the results of OOD detection
7. Configs:
   1. Base-RCNN-FPN.yaml: base FRCNN-FPN model definition config
   2. (dataset)-trained.yaml: For loading the complete pre-trained model checkpoints, based on the training dataset
   3. finetune_(dataset)_trained.yaml: Config file for fine-tuning box and loading the said checkpoint
