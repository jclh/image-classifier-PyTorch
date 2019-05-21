# PROGRAMMER: JC Lopez  
# DATE CREATED: 08/09/2018
# REVISED DATE: 08/24/2018
# PURPOSE: Utility functions for predict.py


# Imports python modules
import os
import random
import argparse
import json
import torch
import numpy as np


def get_input_args():
    """Retrieve and parse command line arguments defined using argparse 
    module. Return arguments as an ArgumentParser object. 

    Five command line arguements are created:
        1. input (str): Path to image
        2. checkpoint (str): Path to checkpoint
        3. top_k (int): Number of most likely classes (default: 1)
        4. category_names (str): Use mapping of categories to real names 
            (default: 'cat_to_name.json')
        5. gpu (bool): Use GPU for training(default: True)
    
    Args:
        None
    Returns:
        parse_args: Container with the command line arguments  
    
    """
    # Create Argument Parser object named parser
    parser = argparse.ArgumentParser()
    
    # Argument 1: Path to image file (Non-optional)
    parser.add_argument('input', type=str, default=None, 
                        help='Path to image file (Non-optional)')
    # Argument 2: Path to checkpoint file (Non-optional)
    parser.add_argument('checkpoint', type=str, default=None, 
                        help='Path to checkpoint file (Non-optional)')
    # Argument 3: Number of most likely classes
    parser.add_argument('--top_k', type=int, default=3, 
                        help = 'Number of most likely classes')
    # Argument 4: Mapping of categories to real names
    parser.add_argument('--category_names', type=str, 
                        default='cat_to_name.json', 
                        help='Mapping of categories to real names')
    # Argument 5: Use GPU for training
    parser.add_argument('--gpu', type=bool, default=True, 
                        help='Use GPU for training')
    
    return parser.parse_args()


def print_input_args(in_args):
    """Print command line arguments

    Args:
        in_args (argparse.ArgumentParser)
 
    """
    print("\nCommand line arguments:",
          "\n    input = ", in_args.input, 
          "\n    checkpoint = ", in_args.checkpoint, 
          "\n    top_k = ", in_args.top_k, 
          "\n    category_names = ", in_args.category_names, 
          "\n    gpu = ", in_args.gpu, 
          "\n")


def class_to_name(filename):
    """Load the datasets and define the dataloaders
    
    Args:
        subdirs_dict (dict): Paths to dataset directories 
        transforms_dict (dict): Data transforms
    Returns:
        class_to_name (dict): Class as key and name as value

    """
    # Load the datasets with ImageFolder
    with open(filename, 'r') as f:
        class_to_name = json.load(f)
    
    return class_to_name


def load_checkpoint(path_checkpoint):
    """Load the model from a saved checkpoint

    Args:
        path_checkpoint (str): Path to checkpoint 
    Returns:
         model (dict): Trained model and other data saved in checkpoint 

    """
    checkpoint = torch.load(path_checkpoint)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    return model


def process_image(pil_image):
    """Scale, crop, and normalize a PIL image for a PyTorch model
    
    Args:
        pil_image (PIL image)
    Returns:
         norm_image (numpy.array)
    
    """
    # Resize image so that shortest side is 256 pixels, keeping ratio
    img_size = pil_image.size
    ratio = max(img_size) / min(img_size)
    new_size = [0,0]
    short = img_size.index(min(img_size))
    long = 1 - short 
    new_size[short] = 255
    new_size[long] = int(255 * ratio)
    new_size = tuple(new_size)
    pil_image = pil_image.resize(size=new_size)

    # Crop out the center 224x224 portion of the image
    gap_x = int((new_size[0] - 224) / 2)
    gap_y = int((new_size[1] - 224) / 2)
    crop_box = (gap_x, gap_y, gap_x + 224, gap_y + 224) 
    pil_image = pil_image.crop(box=crop_box)

    # Re-encode image color channels
    np_image = np.array(pil_image) / 255

    # Normalize image accordingly to the same statistics used to train 
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    norm_image = (np_image - mean) / std

    # Reorder dimensions of NumPy array so it matches PyTorch's input
    norm_image = norm_image.transpose((2, 0, 1))
    
    return norm_image

def image_to_tensor(np_image):
    """Resize NumPy array to match dataloader output size and transform
    to torch tensor 

    Args:
        np_image (numpy.array): Output of process_image(pil_image)
    Returns:
        norm_image (torch)

    """
    # Resize array to match dataloader output size
    np_image = np.resize(np_image,(1, 3, 224, 224))
    
    # NumPy to torch
    img_tensor = torch.from_numpy(np_image)
    img_tensor = img_tensor.type(torch.FloatTensor)
    
    return img_tensor

def random_test_img(test_dir):
    """Select a random test image

    Args:
        test_dir (str): Path to directory with test images
    Returns:
        path_image (str): Path to random test image
    """
    random_class = random.choice(os.listdir(test_dir))
    class_dir = test_dir + str(random_class) + '/'
    random_image = random.choice(os.listdir(class_dir))
    path_image = class_dir + random_image
    
    return path_image