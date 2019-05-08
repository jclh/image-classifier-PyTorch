# PROGRAMMER: JC Lopez  
# DATE CREATED: 08/09/2018
# REVISED DATE: 08/24/2018
# PURPOSE: Utility functions for predict.py
##
# Imports python modules
import os
import random
import argparse
import json
import torch
import numpy as np

def get_input_args():
    """
    Retrieves and parses the command line arguments created and defined using
    the argparse module. This function returns these arguments as an
    ArgumentParser object. 
     5 command line arguements are created:
       input - Path to image (Non-optional)
       checkpoint - Path to checkpoint (Non-optional)
       top_k - Number of most likely classes(default-
              1)
       category_names - Use mapping of categories to real names(default-
              'cat_to_name.json')
       gpu - Use GPU for training(default-
              True)
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    # Creates Argument Parser object named parser
    parser = argparse.ArgumentParser()
    # Argument 1: Path to image file (Non-optional)
    parser.add_argument('input', type = str, default = None, 
                        help = 'Path to image file (Non-optional)')
    # Argument 2: Path to checkpoint file (Non-optional)
    parser.add_argument('checkpoint', type = str, default = None, 
                        help = 'Path to checkpoint file (Non-optional)')
    # Argument 3: Number of most likely classes
    parser.add_argument('--top_k', type = int, default = 3, 
                        help = 'Number of most likely classes')
    # Argument 4: Mapping of categories to real names
    parser.add_argument('--category_names', type = str, default = 'cat_to_name.json', 
                        help = 'Mapping of categories to real names')
    # Argument 5: Use GPU for training
    parser.add_argument('--gpu', type = bool, default = True, 
                        help = 'Use GPU for training')
    return parser.parse_args()

def print_input_args(in_args):
    """
    Prints command line arguments
    Parameters:
     Parser - argparse.ArgumentParser()
    Returns:
     None - Prints command line arguments  
    """
    print("\nCommand line arguments:",
          "\n    input = ", in_args.input, 
          "\n    checkpoint = ", in_args.checkpoint, 
          "\n    top_k = ", in_args.top_k, 
          "\n    category_names = ", in_args.category_names, 
          "\n    gpu = ", in_args.gpu, 
          "\n")

def class_to_name(filename):
    """
    Loads the datasets and defines the dataloaders
    Parameters:
     subdirs_dict - dict with paths to dataset directories 
     transforms_dict - dict with data transforms
    Returns:
         class_to_name - dictionary with class as key 
                                and name as value
    """
    # Load the datasets with ImageFolder
    with open(filename, 'r') as f:
        class_to_name = json.load(f)
    return class_to_name

def load_checkpoint(path_checkpoint):
    """
    Loads the model from a saved checkpoint
    Parameters:
        path_checkpoint - Path to checkpoint 
    Returns:
         model - trained model saved in checkpoint 
    """
    checkpoint = torch.load(path_checkpoint)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model

def process_image(pil_image):
    """
    Scales, crops, and normalizes a PIL image for a PyTorch model
    Parameters:
        pil_image - PIL image 
    Returns:
         norm_image - returns an Numpy array
    """
    img_size = pil_image.size
    ratio = max(img_size) / min(img_size)
    new_size = [0,0]
    short = img_size.index(min(img_size))
    long = 1 - short 
    new_size[short] = 255
    new_size[long] = int(255 * ratio)
    new_size = tuple(new_size)
    pil_image = pil_image.resize(size=new_size)
    gap_x = int((new_size[0] - 224) / 2)
    gap_y = int((new_size[1] - 224) / 2)
    crop_box = (gap_x, gap_y, gap_x + 224, gap_y + 224) 
    pil_image = pil_image.crop(box=crop_box)
    np_image = np.array(pil_image) / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    norm_image = (np_image - mean) / std
    norm_image = norm_image.transpose((2, 0, 1))
    return norm_image

def image_to_tensor(np_image):
    """
    Resizes np array to match dataloader output size
    and transforms to torch tensor 
    Parameters:
        np_image - Numpy array-output of process_image(pil_image)
    Returns:
         norm_image - tensor compatible with model
    """
    # resize array to match dataloader output size
    np_image = np.resize(np_image,(1, 3, 224, 224))
    # numpy to torch
    img_tensor = torch.from_numpy(np_image)
    img_tensor = img_tensor.type(torch.FloatTensor)
    return img_tensor

def random_test_img(test_dir):
    """
    Parameters:
        test_dir - Path to directory with test images
    Returns:
        path_image - Path to random test image
    """
    random_class = random.choice(os.listdir(test_dir))
    class_dir = test_dir + str(random_class) + '/'
    random_image = random.choice(os.listdir(class_dir))
    path_image = class_dir + random_image
    return path_image