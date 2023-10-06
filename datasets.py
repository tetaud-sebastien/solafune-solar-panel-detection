import numpy as np
import cv2
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms, utils
from labels import *
from loguru import logger


def center_crop(img, dim):
    """
    Center crop an image.
    This function crops an image by taking a rectangular region from the center of the image. 
    The size of the rectangular region is determined by the input dimensions.
    Args:
        img (ndarray): The image to be cropped. The image should be a 2D numpy array.
        dim (tuple): A tuple of integers representing the width and height of the crop window.
    Returns:
        ndarray: The center-cropped image as a 2D numpy array.
    Example:
        >>> import numpy as np
        >>> img = np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8)
        >>> cropped_img = center_crop(img, (50, 50))
    """
    width, height = img.shape[1], img.shape[0]
    # process crop width and height for max available dimension
    crop_width = dim[0] if dim[0] < img.shape[1] else img.shape[1]
    crop_height = dim[1] if dim[1] < img.shape[0] else img.shape[0]
    mid_x, mid_y = int(width / 2), int(height / 2)
    cw2, ch2 = int(crop_width / 2), int(crop_height / 2)
    crop_img = img[mid_y - ch2:mid_y + ch2, mid_x - cw2:mid_x + cw2]
    return np.ascontiguousarray(crop_img)


class CustomDataAugmentation():
    """
    Custom data augmentation class for adding Gaussian noise to an image.
    """

    def __init__(self, image, mask, mean=0, std_dev=0.1):
        """
        Initialize the augmentation class.

        Args:
            image (numpy.ndarray): The input image (numpy array).
            mask (numpy.ndarray): The input mask (numpy array).
            mean (float): Mean of the Gaussian distribution (default is 0).
            std_dev (float): Standard deviation of the Gaussian distribution (default is 0.1).
        """
        self.image = image
        self.mask = mask
        self.mean = mean
        self.std_dev = std_dev

    def add_gaussian_noise(self):
        """
        Add Gaussian noise to the image.

        Returns:
            numpy.ndarray: Image with added Gaussian noise.
        """
        # Generate Gaussian noise
        gaussian_noise = np.random.normal(self.mean, self.std_dev, self.image.shape).astype(np.uint8)
    
        # Add noise to the image
        noisy_image = cv2.add(self.image, gaussian_noise)
    
        # Clip the values to stay within the valid image range (0-255)
        noisy_image = np.clip(noisy_image, 0, 255)
    
        return noisy_image


def get_random_crop(image, mask, crop_width=256, crop_height=256):
    """
    Get a random crop from the image and mask.

    Args:
        image (numpy.ndarray): The input image (numpy array).
        mask (numpy.ndarray): The input mask (numpy array).
        crop_width (int): Width of the crop (default is 256).
        crop_height (int): Height of the crop (default is 256).

    Returns:
        tuple: A tuple containing the cropped image and mask as numpy arrays.
    """
    max_x = mask.shape[1] - crop_width
    max_y = mask.shape[0] - crop_height
    x = np.random.randint(0, max_x)
    y = np.random.randint(0, max_y)
    crop_mask = mask[y: y + crop_height, x: x + crop_width]
    crop_image = image[y: y + crop_height, x: x + crop_width, :]
    return crop_image, crop_mask


class TrainDataset(Dataset):
    """
    Custom training dataset class.
    """

    def __init__(self, df_path, transforms=None):
        """
        Initialize the training dataset.

        Args:
            df_path (DataFrame): A DataFrame containing file paths and dataset information.
            transforms (callable): A function/transform to apply to the data (default is None).
        """
        self.df_path = df_path
        self.transforms = transforms

    def __getitem__(self, index):

        img_path = self.df_path.rgb_path.iloc[index]
        image = cv2.imread(img_path)
        image = cv2.resize(image, [image.shape[1]//4, image.shape[0]//4])
        # image = cv2.resize(image, [256, 256])


        if self.df_path.dataset.iloc[index] == "KITTY":
            mask_path = self.df_path.mask_path.iloc[index]
            mask = cv2.imread(mask_path)
            mask = cv2.resize(mask, [256, 256])
            road_label = np.array([255, 0, 255])
            mask = np.all(mask == road_label, axis=-1).astype(np.uint8)


        if self.df_path.dataset.iloc[index] == "FREIBURG":
            mask_path = self.df_path.mask_path.iloc[index]
            mask = cv2.imread(mask_path)
            mask = cv2.resize(mask, [256, 256])
            road_label = np.array([170, 170, 170])
            mask = np.all(mask == road_label, axis=-1).astype(np.uint8)



        if self.df_path.dataset.iloc[index] == "ORFD":
            mask_path = self.df_path.mask_path.iloc[index]
            mask = cv2.imread(mask_path)
            mask = cv2.resize(mask, [256, 256])
            road_label = np.array([255, 255, 255])
            mask = np.all(mask == road_label, axis=-1).astype(np.uint8)

        if self.df_path.dataset.iloc[index] == "CITYSCAPES":
            
            mask_path = self.df_path.mask_path.iloc[index]
            mask = cv2.imread(mask_path)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            
            mask = cv2.resize(mask, [mask.shape[1]//4, mask.shape[0]//4])
            # mask = cv2.resize(mask, [256, 256])
            
            label = [i for i in range (7,30)]

            list_class_mask = []
            for i in label:
            
                label_mask = np.all(mask == np.array(labels[i].color), axis=-1).astype(np.uint8)
                list_class_mask.append(label_mask)

            mask = np.array(list_class_mask)
            mask = torch.from_numpy(mask)
            






        if self.transforms:

            data_augmentation = CustomDataAugmentation(image=image, mask=mask, mean=0, std_dev=0.1)
            bright_value = np.random.randint(0, 70)
            choices = ['decrease_brightness', 'increase_brightness', 'no_aug']
            aug_choice = random.choice(choices)

            if aug_choice == "decrease_brightness":
                bright = np.ones(image.shape, dtype="uint8") * bright_value
                image = cv2.subtract(image, bright)
            elif aug_choice == "increase_brightness":
                bright = np.ones(image.shape, dtype="uint8") * bright_value
                image = cv2.add(image, bright)
            elif aug_choice == "no_aug":
                pass  # No augmentation

            image = data_augmentation.add_gaussian_noise()
            choices = ['vhflip', 'no_aug']
            aug_choice = random.choice(choices)

            if aug_choice == 'hflip':
                mask = cv2.flip(mask, 1)
                image = cv2.flip(image, 1)
            elif aug_choice == 'no_aug':
                pass  # No augmentation

            image = image / 255.0
            image = torch.from_numpy(image).float()
            image = torch.permute(image, (2, 0, 1))
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, [256, 256])
            image = image / 255.0
            image = torch.from_numpy(image).float()
            image = torch.permute(image, (2, 0, 1))
            mask = cv2.resize(mask, [256, 256])

        # return image, mask, depth
        return image, mask

    def __len__(self):
        return len(self.df_path)


class EvalDataset(Dataset):


    def __init__(self, df_path ,transforms=None):
        
        self.df_path = df_path
        self.transforms = transforms

    def __getitem__(self, index):

        img_path = self.df_path.rgb_path.iloc[index]
        #INPUT
        image = cv2.imread(img_path)
        # image = cv2.resize(image,[256, 256])
        image = cv2.resize(image, [image.shape[1]//4, image.shape[0]//4])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)/255.0
        image = torch.from_numpy(image).float()
        image = torch.permute(image, (2, 0, 1))

        if self.df_path.dataset.iloc[index] == "KITTY":

            mask_path = self.df_path.mask_path.iloc[index]
            mask = cv2.imread(mask_path)
            mask = cv2.resize(mask,[256,256])
            road_label = np.array([255, 0, 255])
            binary_mask = np.all(mask == road_label, axis=-1).astype(np.uint8)

        if self.df_path.dataset.iloc[index] == "FREIBURG":
            mask_path = self.df_path.mask_path.iloc[index]
            mask = cv2.imread(mask_path)
            mask = cv2.resize(mask, [256, 256])
            road_label = np.array([170, 170, 170])
            mask = np.all(mask == road_label, axis=-1).astype(np.uint8)

            
        if self.df_path.dataset.iloc[index] == "ORFD":
            mask_path = self.df_path.mask_path.iloc[index]
            mask = cv2.imread(mask_path)
            mask = cv2.resize(mask, [256, 256])
            road_label = np.array([255, 255, 255])
            mask = np.all(mask == road_label, axis=-1).astype(np.uint8)

        if self.df_path.dataset.iloc[index] == "CITYSCAPES":
            
            
            mask_path = self.df_path.mask_path.iloc[index]
            mask = cv2.imread(mask_path)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            mask = cv2.resize(mask, [mask.shape[1]//4, mask.shape[0]//4])
            # mask = cv2.resize(mask, [256, 256])
            label = [i for i in range (7,30)]

            list_class_mask = []
            for i in label:
            
            
                
                label_mask = np.all(mask == np.array(labels[i].color), axis=-1).astype(np.uint8)
                list_class_mask.append(label_mask)

            mask = np.array(list_class_mask)
            mask = torch.from_numpy(mask)


        return image, mask


    def __len__(self):
        return len(self.df_path)

    