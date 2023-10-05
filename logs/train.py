import os 
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import confusion_matrix
from torchvision.transforms import ToTensor
from utils import * 

from loguru import logger
from labels import *

MAX_TENSORBOARD_IMAGES = 5  # Adjust as needed


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def train_log(step, loss, tensorboard_writer, name, **kwargs):


    
    if step % 100 == 0:
        loss = loss.detach().cpu().numpy()
        if name=='Validation':
           
            print(loss.detach().cpu().numpy())
        # print(kwargs)
        # current_lr = kwargs['current_lr']
        # print('step={}, loss: {:.12f}'.format(step, loss))
        if tensorboard_writer is not None:
            tensorboard_writer.add_scalar(name+'/loss', loss, step)
            # tensorboard_writer.add_scalar(name+'/learning_rate', current_lr, step)


def val_log(step, images_inputs, seg_targets, seg_preds,
            tensorboard_writer, name, prediction_dir, **kwargs):
    """
    Logs validation-related information and visualizations to TensorBoard.

    Args:
        step (int): Current step or epoch.
        input_batch (torch.Tensor): Input batch of images.
        target (torch.Tensor): True target labels.
        output_batch (torch.Tensor): Model's predicted output batch.
        labels (dict): Dictionary mapping class indices to class labels.
        confusion_matrix (np.ndarray): Confusion matrix for evaluation.
        tensorboard_writer (SummaryWriter): TensorBoard writer for logging.
        name (str): Name prefix for the logs.
        prediction_dir (str): Directory to save prediction visualizations.
        **kwargs: Additional keyword arguments (e.g., loss).

    Returns:
        None 
    """

    if not os.path.exists(prediction_dir):
        print("Create prediction directory...")
        os.makedirs(prediction_dir, exist_ok=True)
    now = datetime.now()
    time = now.strftime("%m_%d_%Y_%H_%M_%S")

    # Extract loss value from kwargs
    loss = kwargs['loss'].detach().cpu().numpy()

    if tensorboard_writer is not None:
        tensorboard_writer.add_scalar(name + '/loss', loss, step)

    # Prepare images and predictions for TensorBoard visualization
    images_inputs = torch.permute(images_inputs, (0, 2, 3, 1))
    images = images_inputs.detach().cpu().numpy()
    seg_targets = seg_targets.detach().cpu().numpy()

    if step % 400 == 0:
        for i in range(min(images.shape[0], MAX_TENSORBOARD_IMAGES)):

            image = images[i]
            seg_pred = seg_preds[i]
            mask = seg_targets[i]
            
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

            # pred_rgb = cv2.cvtColor(pred_rgb, cv2.COLOR_BGR2RGB)
            image_tensorboard = np.hstack((image, pred_rgb))
            # # Add the input image to TensorBoard
            tensorboard_writer.add_image(f'{name}/input', image_tensorboard, dataformats='HWC')
            # # Save the image with prediction label to the prediction directory
            now = datetime.now()
            time = now.strftime("%m_%d_%Y_%H_%M_%S")
            filename = os.path.join(prediction_dir, f'{time}_{i}.png')
            cv2.imwrite(filename, image_tensorboard)
