# PROGRAMMER: JC Lopez  
# DATE CREATED: 08/09/2018
# REVISED DATE: 08/24/2018
# PURPOSE: Utility functions for train.py
##
# Imports python modules
import os
import argparse
import torch
from torchvision import transforms, datasets
import json

def get_input_args():
    """
    Retrieves and parses the command line arguments created and defined using
    the argparse module. This function returns these arguments as an
    ArgumentParser object. 
     7 command line arguements are created:
       data_dir - Path to the directory with images (Non-optional)(default-
              'flowers/'
       arch - CNN model architecture to use for image classification(default-
              vgg11
       save_dir - Directory to save checkpoints(default-
              'checkpoints/'
       learning_rate - Model learning rate(default-
              0.001
       hidden_units - Units in hidden layer pre-classifier(default-
              4096
       epochs - Number of passes of the training data(default-
              5
       gpu - Use GPU for training(default-
              True
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    # Creates Argument Parser object named parser
    parser = argparse.ArgumentParser()
    # Argument 1: Path to the directory with images (Non-optional)
    parser.add_argument('data_dir', type = str, 
                        default = None, 
                        help = 'Path to the directory with images (Non-optional)')
    # Argument 2: CNN model architecture to use for image classification
    parser.add_argument('--arch', type = str, default = 'vgg11', 
                        help = 'CNN model architecture to use for image classification')
    # Argument 3: Directory to save checkpoints
    parser.add_argument('--save_dir', type = str, default = None, 
                        help = 'Directory to save checkpoints')
    # Argument 4: Model learning rate
    parser.add_argument('--learning_rate', type = float, default = 0.001, 
                        help = 'Model learning rate')
    # Argument 5: Units in hidden layer pre-classifier
    parser.add_argument('--hidden_units', type = int, default = 512, 
                        help = 'Units in hidden layer pre-classifier')
    # Argument 6: Number of passes of the training data
    parser.add_argument('--epochs', type = int, default = 5, 
                        help = 'Number of passes of the training data')
    # Argument 7: Use GPU for training
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
          "\n    dir = ", in_args.data_dir, 
          "\n    arch = ", in_args.arch, 
          "\n    save_dir = ", in_args.save_dir, 
          "\n    learning_rate = ", in_args.learning_rate, 
          "\n    hidden_units = ", in_args.hidden_units, 
          "\n    epochs = ", in_args.epochs, 
          "\n    gpu = ", in_args.gpu, 
          "\n")

def data_subdirs(data_dir):
    """
    Returns dictionary with data sub-directories  
    Parameters:
     data_dir - string
    Returns:
     subdirs_dict - dictionary with data sub-directories 
    """
    subdirs_dict = dict()
    subdirs_dict['train'] = data_dir + 'train/'
    subdirs_dict['valid'] = data_dir + 'valid/'
    subdirs_dict['test'] = data_dir + 'test/'
    return subdirs_dict

def data_transforms():
    """
    Returns dictionary with data transforms for the 
    training, validation, and testing sets
    Parameters:
     None -
    Returns:
     transforms_dict - dictionary with data transforms for the 
                        training, validation, and testing sets
    """
    transforms_dict = dict()
    transforms_dict['train'] = transforms.Compose([
                        transforms.RandomRotation(30),
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], 
                                             [0.229, 0.224, 0.225])
                        ])
    transforms_dict['valid'] = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], 
                                             [0.229, 0.224, 0.225])
                        ])
    transforms_dict['test'] = transforms.Compose([transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], 
                                             [0.229, 0.224, 0.225])
                         ])
    return transforms_dict

def data_loaders(subdirs_dict, transforms_dict):
    """
    Loads the datasets and defines the dataloaders
    Parameters:
     subdirs_dict - dict with paths to dataset directories 
     transforms_dict - dict with data transforms
    Returns:
         dataloader_dict - dictionary with dataloaders
         class_to_idx_dict - mapping from class number to index in tensors
    """
    # Load the datasets with ImageFolder
    datasets_dict = dict()
    datasets_dict['train'] = datasets.ImageFolder(
                                  subdirs_dict['train'], 
                                  transform=transforms_dict['train'])
    datasets_dict['valid'] = datasets.ImageFolder(
                                  subdirs_dict['valid'], 
                                  transform=transforms_dict['valid'])
    datasets_dict['test'] = datasets.ImageFolder(
                                  subdirs_dict['test'], 
                                  transform=transforms_dict['test'])
    
    # Gets the mapping from class number in data folders 
    # to index of labels tensor
    class_to_idx_dict = datasets_dict['train'].class_to_idx
    
    # Using the image datasets and the trainforms, define the dataloaders
    dataloaders_dict = dict()
    dataloaders_dict['train'] = torch.utils.data.DataLoader(
                            datasets_dict['train'], 
                            batch_size=32, shuffle=True)
    dataloaders_dict['valid'] = torch.utils.data.DataLoader(
                            datasets_dict['train'], 
                            batch_size=32, shuffle=True)
    dataloaders_dict['test'] = torch.utils.data.DataLoader(
                            datasets_dict['train'], 
                            batch_size=32, shuffle=True)
    return dataloaders_dict, class_to_idx_dict