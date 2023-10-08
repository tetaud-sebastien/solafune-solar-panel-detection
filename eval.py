import argparse
import os
import datetime
import time
import torch
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# from models import Unet
from resunet import Unet

from torch.utils.data.dataloader import DataLoader
from datasets import TrainDataset, EvalDataset
import cv2
from utils import *
from evaluate_depth import *


import json
from torch import nn
from collections import OrderedDict

from loguru import logger
import segmentation_models_pytorch as smp


logger.add("out.log", backtrace=False, diagnose=True)


def load_model(model_path, device):
    """
    Load the trained model from the given path.

    Args:
        model_path (str): Path of the trained model.

    Returns:
        model (nn.Module): The loaded model.

    """
    # model = Unet()

    import segmentation_models_pytorch as smp

    model = smp.Unet(
        encoder_name="vgg19",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        # use `imagenet` pre-trained weights for encoder initialization
        # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        in_channels=3,
        # model output channels (number of classes in your dataset)
        classes=23,
    )

    if device.type == 'cpu':
        checkpoint = torch.load(model_path, map_location=device)
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            name = k[7:] #remove 'module'
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        model.eval()

        logger.info(model.eval())

    else:
        
        logger.info("==> Loading checkpoint '{}'".format(model_path))
        checkpoint = torch.load(model_path, map_location=device)
        model = torch.nn.DataParallel(model)
        model.load_state_dict(checkpoint)
        model.eval()
        model.cuda()
    
    return model


def main(args):
    """
    Main function to test the trained model on the given test data.

    Args:
        config (dict): The configuration dictionary for the test.
        args (argparse.Namespace): The command-line arguments.

    """
    
    model_path = args.checkpoint_path
    source = args.input
    hardware = args.hardware
    
    if hardware=='cuda':
        device = torch.device(hardware)

        try:
            model = load_model(model_path, device)
    
        except Exception as e:
            logger.exception("An error occurred: {}", e)
        
        model.to(hardware)
        if len(str(hardware))>1:
            model= nn.DataParallel(model)
        
        logger.info("------")
        logger.info("Model run on: {}".format(hardware))
        logger.info("------")
       

    else:
        device = torch.device(hardware)
        model = load_model(model_path, device)
        model.to(hardware)
        logger.info("Model run on: {}".format(hardware))
        
    prediction_dir = "evaluation"
    if not os.path.exists("evaluation"):
        # console.print("Create prediction directory...", style="bold green")
        logger.info("Create prediction directory...")
        os.makedirs(prediction_dir, exist_ok=True)

    date = datetime.datetime.now()
    date = date.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = os.path.join(prediction_dir, '{}'.format(date))
    os.makedirs(output_dir, exist_ok=True)
        

    # Read input
    input = args.input
    
    test_path = pd.read_csv(input)
    # test_path = test_path[:10]
    logger.info("Number of Test data {0:d}".format(len(test_path)))    
    logger.info("------")
    eval_dataset = EvalDataset(df_path= test_path, transforms=None)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1, shuffle=False)

    iou_metrics = []
    f1_score_metrics = []
    f2_score_metrics = []
    accuracy_metrics = []
    recall_metrics = []
    metrics_dict = {}

    for index, data in enumerate(eval_dataloader):
        # for data in eval_dataloader:
        image, seg_targets = data
        
        image = image.to(device)
        seg_targets = seg_targets.to(device)

        with torch.no_grad():
            
            seg_preds = model(image)
            logger.info(seg_preds.shape)

        
        tp, fp, fn, tn = smp.metrics.get_stats(seg_preds, seg_targets, mode='multilabel', threshold=0.5)
        iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        
        f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
        f2_score = smp.metrics.fbeta_score(tp, fp, fn, tn, beta=2, reduction="micro")
        accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro")
        recall = smp.metrics.recall(tp, fp, fn, tn, reduction="micro-imagewise")

        iou_metrics.append(float(iou_score.detach().cpu().numpy()))
        f1_score_metrics.append(float(f1_score.detach().cpu().numpy()))
        f2_score_metrics.append(float(f2_score.detach().cpu().numpy()))
        accuracy_metrics.append(float(accuracy.detach().cpu().numpy()))
        recall_metrics.append(float(recall.detach().cpu().numpy()))


        seg_pred = seg_preds[0,:,:,:]

        image = torch.permute(image, (0, 2, 3, 1))
        image = image.detach().cpu().numpy()
        image = image[0,:,:,:]


        class_probs = torch.softmax(seg_pred, dim=0)
        mask_prediction = torch.argmax(class_probs, dim=0).cpu().numpy()
        image = ((image - image.min()) / (image.max() - image.min())) * 255
        image = image.astype(np.uint8)


        label_ids_to_use = [i for i in range (7,30)]
        # Convert label IDs to indices for easy access
        label_indices = {label.id: idx for idx, label in enumerate(labels)}
        class_colors_dict = {label.id: label.color for label in labels if label.id in label_ids_to_use}

        pred_rgb = np.zeros((mask_prediction.shape[0], mask_prediction.shape[1], 3), dtype=np.uint8)
        for c, color in enumerate(class_colors_dict.items()):
            
            # reverse color to have identic color mask as cityscapes
            color_class = np.array(color[1])

            color_class[0], color_class[-1] = color_class[-1], color_class[0]
            color_class = tuple(color_class)
            pred_rgb[mask_prediction == c] = color_class

        image_output = np.hstack((image, pred_rgb))

        image_filename = os.path.join(output_dir, str(index)+".png")
        cv2.imwrite(image_filename, image_output)





        

    




    metrics_dict = {"IoU": np.mean(iou_metrics), "F1": np.mean(f1_score_metrics), "F2": np.mean(f2_score_metrics),
                               "Accuracy": np.mean(recall_metrics), "Recall": np.mean(recall_metrics)}

    report_filename = os.path.join(output_dir, "test_report.json")
    with open(report_filename, 'w') as json_file:
        json.dump(metrics_dict, json_file, indent=4)

    logger.info("------")
    logger.info("Evaluation Done")
    logger.info("------")
    #     IoU = intersection_over_union(predictions=seg_preds, targets=seg_targets,threshold=threshold)
    #     IoU_metrics.append(IoU)

    #     seg_preds = seg_preds.detach().cpu().numpy()
    #     seg_targets = seg_targets.detach().cpu().numpy()

    #     image = images_inputs[0,:,:,:].detach().cpu().numpy()
    #     class_probs = torch.softmax(seg_pred, dim=0)
    #         mask_prediction = torch.argmax(class_probs, dim=0).cpu().numpy()
    #         image = ((image - image.min()) / (image.max() - image.min())) * 255
    #         image = image.astype(np.uint8)

    #         label_ids_to_use = [i for i in range (7,30)]
    #         # Convert label IDs to indices for easy access
    #         label_indices = {label.id: idx for idx, label in enumerate(labels)}
    #         class_colors_dict = {label.id: label.color for label in labels if label.id in label_ids_to_use}
            
    #         pred_rgb = np.zeros((mask_prediction.shape[0], mask_prediction.shape[1], 3), dtype=np.uint8)
    #         for c, color in enumerate(class_colors_dict.items()):
                
    #             # reverse color to have identic color mask as cityscapes
    #             color_class = np.array(color[1])

    #             color_class[0], color_class[-1] = color_class[-1], color_class[0]
    #             color_class = tuple(color_class)
    #             pred_rgb[mask_prediction == c] = color_class
    #     image_seg = np.copy(image)
    #     # boolean indexing and assignment based on mask
    #     image_seg[(mask==255).all(-1)] = [0,255,0]
    #     image_seg = cv2.addWeighted(image_seg, 0.3, image, 0.7, 0, image_seg)
    #     image_seg = cv2.resize(image_seg, (512, 512))
       
    #     result = np.hstack([image_seg])

    #     image_filename = os.path.join(output_dir, str(index)+".png")
    #     cv2.imwrite(image_filename, result)


    # IoU_metrics = np.concatenate(IoU_metrics)
    # mean_iou = float(IoU_metrics.mean())
    
    # metrics_dict = {"IoU":mean_iou}

    # logger.debug(metrics_dict)
    
    # report_filename = os.path.join(output_dir, "test_report.json")
    # with open(report_filename, 'w') as json_file:
    #     json.dump(metrics_dict, json_file, indent=4)

    # logger.info("------")
    # logger.info("Evaluation Done")
    # logger.info("------")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='MetricLeReS inference (video/webcam)', fromfile_prefix_chars='@')

    # Input
    parser.add_argument('-i', '--input', help='csv path', type=str, default='0')
    parser.add_argument('-c', '--checkpoint_path',type=str,   help='path to a checkpoint to load', default='')
    parser.add_argument("-d", "--hardware", type=str, help="device - gpu/cpu", default='cuda')
    args = parser.parse_args()

    main(args)
