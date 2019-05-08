# PROGRAMMER: JC Lopez  
# DATE CREATED: 08/09/2018
# REVISED DATE: 08/24/2018
# PURPOSE: Predict flower name from image along with the probability of that name.
# BASIC USAGE:
#      python predict.py <path to image> <checkpoint>
#             --top_k <number of most likely classes>
#             --category_names <mapping of categories to real names>
#             --gpu
#   Example basic usage:
#    python predict.py in_image checkpoint --top_k 3 --gpu
#                       --category_names cat_to_name.json
##
# Imports python modules
from time import time, sleep
from utility_fs_predict import *
from model_functions import *
from PIL import Image

def main():    
    # Collect start time
    start_time = time()
 
    # Define get_input_args() function to create 
    # and retrieve command line arguments
    in_args = get_input_args()
    print_input_args(in_args)
    
    # If in_args.input == random_test, pick random test image
    if in_args.input == 'random_test':
        image_path = random_test_img(test_dir='flowers/test/')
    else:
        image_path = in_args.input
        
    # Load model from checkpoint
    model = load_checkpoint(in_args.checkpoint)  
    
    # Open image as PIL object
    pil_image = Image.open(image_path)
    
    # Process PIL image to normalized Numpy array
    np_image = process_image(pil_image)
    
    # Resize array to match dataloader output size
    # and return torch tensor 
    img_tensor = image_to_tensor(np_image)

    # Resize array to match dataloader output size
    # and return torch tensor     
    top_probs, classes = predict(model, 
                                 img_tensor, 
                                 in_args.top_k, 
                                 in_args.gpu)
    
    # Import dictionary of keys = class number (as in data folders)
    # and values = flower names (in words)
    class_name_dict = class_to_name(filename=in_args.category_names)
    flower_names = [class_name_dict[key] for key in classes]
    
    print('\n Filepath to image: ', image_path, '\n',
          '\n  Classes: ', classes,
          '\n  Flower names: ', flower_names,
          '\n  Probabilities: ', top_probs)    
    
    # Define end_time to measure total program runtime
    end_time = time()
    tot_time = end_time - start_time
    print('\n** Total Elapsed Runtime:', tot_time, 
          '\n')
    
    # Return the flower name and class probability
    return classes, top_probs 
# Call to main function to run the program
if __name__ == "__main__":
    main()