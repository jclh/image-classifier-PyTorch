# Imports python modules
import torch
from torchvision import models
from collections import OrderedDict
from torch import nn, optim


def choose_net_arch(arch):
    """Return pretrained network from torchvision.models

    Args:
        arch (str): User input
    Returns:
        model(torchvision.models): Pre-trained network from 
            torchvision.models

    """
    model = getattr(models, arch)
    return model(pretrained=True)


def build_classifier(model, hidden_units):
    """Return classifier layer for the model
    
    Args:
        model (torchvision.models): Pre-trained model
        hidden_units (int): Number of units in hidden layer (user input)
    Returns:
        classifier (torch.nn.modules.container.Sequential)

    """
    in_features = model.classifier._modules['0'].in_features
    classifier = nn.Sequential(OrderedDict([
        ('dropout1', nn.Dropout(0.5)),
        ('fc1', nn.Linear(in_features, hidden_units)),       
        ('relu', nn.ReLU()),
        ('dropout2', nn.Dropout(0.5)),
        ('fc2', nn.Linear(hidden_units, 102)),
        ('output', nn.LogSoftmax(dim=1))
        ]))
    return classifier


def training(model, trainloader, in_args_epochs, in_args_gpu, 
        in_args_learning_rate):
    """Train the model and print running training loss
    
    Args:
        model(torchvision.models)
        trainloader (torch.utils.data.dataloader.DataLoader)
        in_args_epochs (int): User input
        in_args_gpu (bool): User input
        in_args_learning_rate (float): User input

    """
    if in_args_gpu == True:
        model.to('cuda')
    epochs = in_args_epochs
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), 
                            lr=in_args_learning_rate)
    print_every = 100
    steps = 0
    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1
            if in_args_gpu == True:
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
            optimizer.zero_grad()
            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if steps % print_every == 0:
                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Training Loss: {:.4f}".format(running_loss/print_every))
                running_loss = 0
 

def validation_stats(model, validloader, in_args_gpu):
    """Print running validation loss and total validation accuracy
    
    Args:
        model(torchvision.models)
        validloader (torch.utils.data.dataloader.DataLoader)
        in_args_gpu (bool): User input

    """
    criterion = nn.NLLLoss()
    running_loss = 0
    print_every = 100
    steps = 0
    validation_correct = 0
    validation_total = 0
    with torch.no_grad():
        for data in validloader:
            if steps == 0: print('\n')
            steps += 1
            images, labels = data
            if in_args_gpu == True:
                images, labels = images.to('cuda'), labels.to('cuda')
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            validation_total += labels.size(0)
            validation_correct += (predicted == labels).sum().item()
            if steps % print_every == 0:
                print("Validation Loss: {:.4f}"
                      .format(running_loss/print_every))
                running_loss = 0           
    print('\nValidation accuracy: {1:.1%} \n({0:d} validation images)'
            .format(validation_total, validation_correct / validation_total))


def predict(model, img_tensor, in_args_topk, in_args_gpu):
    """Predict the class(es) of image using trained deep learning model
    
    Args:
        model(torchvision.models)
        img_tensor (torch)
        in_args_topk (int): User input
        in_args_gpu (bool): User input
 
    """
    if in_args_gpu == True:
        model.to('cuda')
        img_tensor = img_tensor.to('cuda')
    # Run model in evaluation mode
    model.eval()
    with torch.no_grad():
        outputs = model(img_tensor)
    # From softmax to probabilities
    probs = torch.exp(outputs.data) 
    # Find topk probabilities and indices 
    top_probs, indices = torch.topk(probs, dim=1, k=in_args_topk) 
    # From torch to numpy to lists
    if in_args_gpu == True:
        top_probs, indices = top_probs.to('cpu'), indices.to('cpu')
    top_probs, indices = top_probs.numpy(), indices.numpy()
    top_probs, indices = top_probs[0].tolist(), indices[0].tolist()
    # Find the class using the indices (reverse dictionary first)
    idx_to_class = {idx: class_ for class_, idx in model.class_to_idx.items()}
    classes = [idx_to_class[i] for i in indices]
    return top_probs, classes