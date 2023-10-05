import argparse
import os
import datetime
import time
import torch
import yaml
import numpy as np
from resunet import Unet
import cv2
from torch import nn
from collections import OrderedDict


def load_model(model_path, device):
    """
    Load the trained model from the given path.

    Args:
        model_path (str): Path of the trained model.

    Returns:
        model (nn.Module): The loaded model.

    """
    model = Unet()

    if device.type == 'cpu':
        checkpoint = torch.load(model_path, map_location=device)
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            name = k[7:] #remove 'module'
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        model.eval()

    else:
        
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))
        model.eval()
        model.cuda()
    
    return model


def main(args, labels):
    """
    Main function to test the trained model on the given test data.

    Args:
        config (dict): The configuration dictionary for the test.
        args (argparse.Namespace): The command-line arguments.

    """
    
    model_path = args.checkpoint_path
    source = args.input
    hardware = args.hardware
    labels = labels['names']
   
    if hardware=='cuda':
        device = torch.device(hardware)
        model = load_model(model_path, device)
        model.to(hardware)
        if len(str(hardware))>1:
            model= nn.DataParallel(model)
        print("------")
        print("Model run on: {}".format(hardware))
        print("------")

    else:
        device = torch.device(hardware)
        model = load_model(model_path, device)
        model.to(hardware)
        print("------")
        print("Model run on: {}".format(hardware))
        print("------")

    prediction_dir = "inference_test"
    if not os.path.exists("inference_test"):
        print("Create prediction directory...")
        os.makedirs(prediction_dir, exist_ok=True)
    # Read input
    input = args.input
    cap = cv2.VideoCapture(eval(input) if input.isnumeric() else input)
    
    print("press q to leave")
    counter = 0
    while(True):
        ret, raw_image = cap.read()

        if not ret:
            break
        
        image_resize = cv2.resize(raw_image, (256,256))
        image = cv2.cvtColor(image_resize, cv2.COLOR_BGR2RGB)/255.0
        image = torch.tensor(image)
        image = torch.permute(image, (2, 0, 1))
        image= image.unsqueeze(0)
        inputs = image.to(device, dtype=torch.float)
        inference_start = datetime.datetime.now()
        with torch.no_grad():
            preds = model(inputs)

        inference_end = datetime.datetime.now()
        inference = (inference_end-inference_start).total_seconds()
        fps = int(1/inference)
        
        preds = preds.detach().cpu().numpy()
        threshold = 0.5
        binary_predictions = (preds > threshold).astype(np.uint8)
        binary_predictions = binary_predictions[0,0,:,:]
        print(binary_predictions)
        

        pred_rgb = np.zeros((binary_predictions.shape[0], binary_predictions.shape[1], 3), dtype=np.uint8)
        pred_rgb[..., 0] = binary_predictions * 255  
        pred_rgb[..., 1] = binary_predictions * 255  
        pred_rgb[..., 2] = binary_predictions * 255


        _, mask = cv2.threshold(pred_rgb, thresh=128, maxval=255, type=cv2.THRESH_BINARY)
        image_seg = np.copy(image_resize)
        image_seg[(mask==255).all(-1)] = [0,255,0]

        image_seg = cv2.addWeighted(image_seg, 0.3, image_resize, 0.7, 0, image_seg)
        image_seg = cv2.resize(image_seg, (512, 512))
        
        
       
        fps_text = f"FPS: {fps}"
        cv2.putText(image_seg, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        # cv2.imshow('on-road/off-road classification', pred_image)
        now = datetime.datetime.now()
        time = now.strftime("%m_%d_%Y_%H_%M_%S")
        filename = os.path.join(prediction_dir, f'{time}_{counter}.png')
        cv2.imwrite(filename, image_seg)
        counter = counter + 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
  

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='MetricLeReS inference (video/webcam)', fromfile_prefix_chars='@')

    # Input
    parser.add_argument('-i', '--input', help='Camera index/Video file path', type=str, default='0')
    parser.add_argument('-c', '--checkpoint_path',type=str,   help='path to a checkpoint to load', default='')
    parser.add_argument("-d", "--hardware", type=str, help="device - gpu/cpu", default='cuda')
    args = parser.parse_args()
    

    with open("labels.yaml", "r") as stream:
        try:
            labels = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    main(args, labels)
