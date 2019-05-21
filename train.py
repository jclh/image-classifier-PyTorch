# PROGRAMMER: JC Lopez  
# DATE CREATED: 08/09/2018
# REVISED DATE: 05/21/2019
# PURPOSE: Train a new network on a dataset and save the model as a 
# checkpoint.
# 
# BASIC USAGE:
#      python train.py <data_directory> 
#             --arch <network architecture>
#             --save_dir <checkpoint directory>
#             --learning_rate <learning rate>
#             --hidden_units <units in hidden layer pre-classifier>
#             --epochs <number of passes of the training data>
#             --gpu
#   Example basic usage:
#    python train.py flowers/ 

# Imports python modules
from time import time, sleep
from os import listdir
import torch

from utility_fs_train import *
from model_functions import *


def main():    
    # Collect start time
    start_time = time()
    
    # Define get_input_args() function to create 
    # and retrieve command line arguments
    in_args = get_input_args()
    print_input_args(in_args)

    # Load the datasets (including transforms) 
    # and define dataloaders
    subdirs_dict = data_subdirs(in_args.data_dir)
    transforms_dict = data_transforms()
    dataloaders_dict, class_to_idx_dict = data_loaders(
        subdirs_dict, transforms_dict)
    
    # Import chosen pretrained network from torchvision.models
    model = choose_net_arch(in_args.arch)
    model.class_to_idx = class_to_idx_dict
    
    # Freeze parameters of pretrained network
    # so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
    
    # Build model classifier 
    model.classifier = build_classifier(model, in_args.hidden_units)
    
    # Train the classifier layers
    training(model, dataloaders_dict['train'], 
             in_args.epochs, in_args.gpu, in_args.learning_rate)
   
    # Run validation and print stats
    validation_stats(model, dataloaders_dict['valid'], in_args.gpu)
    
    # Save model to checkpoint
    checkpoint = {
        'model': model,
        'learn_rate': in_args.learning_rate,
        'epochs': in_args.epochs,
        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx
        }
    
    if in_args.save_dir is None:
        torch.save(checkpoint, 'checkpoint.pth')
    else:
        torch.save(checkpoint, in_args.save_dir + 'checkpoint.pth')            
    
    # Define end_time to measure total program runtime
    end_time = time()
    tot_time = end_time - start_time
    print('\n** Total Elapsed Runtime:', tot_time, '\n')
    
# Call to main function to run the program
if __name__ == "__main__":
    main()