from sklearn.metrics import confusion_matrix
import torch
import numpy as np
from tqdm import tqdm

def train_iter(log_interval, model, device, optimizer, loss_func, data, target):
    '''
    Train the model for a single iteration.
    An iteration is when a single batch of data is passed forward and
    backward through the neural network.
    '''
    data, target = data.to(device), target.to(device)  # Move this batch of data to the specified device.
    optimizer.zero_grad()  # Zero out the old gradients (so we only use new gradients for a new update iteration).
    output = model(data)  # Forward the data through the model.
    loss = loss_func(output, target.float())  # Calculate the loss
    loss.backward()  # Backward the loss and calculate gradients for parameters.
    optimizer.step()  # Update the parameters.
    return loss

def train_epoch(log_interval, model, device, train_loader, optimizer, epoch, loss_func):
    '''
    Train the model for an epoch.
    An epoch is when the entire dataset is passed forward and
    backward through the neural network for once.
    The number of batches in a dataset is equal to number of iterations for one epoch.
    '''
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):  # Iterate through the entire dataset to form an epoch.
        loss = train_iter(log_interval, model, device, optimizer, loss_func, data, target)  # Train for an iteration.
        if batch_idx % log_interval == 0:
            print('\nTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            
def train_network(log_interval,model, device, train_loader, optimizer, loss,epochs=100):
    for epoch in tqdm(range(epochs)):
        train_epoch(log_interval, model, device, train_loader, optimizer, epoch, loss)

def test(model, device, test_loader, loss_func):
    '''
    Testing the model on the entire test set.
    '''
    model.eval()  # Switch the model to evaluation mode, which prevents the dropout behavior.
    test_loss = 0
    tp = 0 # True positives
    fp = 0 # False positives
    fn = 0 # False negatives
    sigmoid_fun = torch.nn.Sigmoid()
    with torch.no_grad():  # Because this is testing and no optimization is required, the gradients are not needed.
        for data, target in tqdm(test_loader):  # Iterate through the entire test set.
            data, target = data.to(device), target.to(device)  # Move this batch of data to the specified device.
            output = model(data)  # Forward the data through the model.
            test_loss += target.size(0)*loss_func(output, target.float()).item()  # Sum up batch loss
            pred = torch.where(sigmoid_fun(output) >= 0.5, 1, 0)  # Get predictions in right format
            tp_temp = (pred*target).sum().item()
            fp_temp = (torch.maximum(pred-target,torch.zeros_like(pred))).sum().item()
            fn_temp = (torch.maximum(target-pred,torch.zeros_like(pred))).sum().item()
            #_, fp_temp, fn_temp, tp_temp = confusion_matrix(target.view_as(pred), pred).ravel() # Get relevant counts for F1 Score
            tp += tp_temp
            fp += fp_temp
            fn += fn_temp 

    test_loss /= len(test_loader.dataset)  # Average the loss on the entire testing set.
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    F1 = (2*precision*recall)/(precision + recall) # Compute final F1 Score

    print('\nTest set results: Average loss: {:.4f}, F1 Score: {:.2f}\n'.format(
        test_loss, F1))