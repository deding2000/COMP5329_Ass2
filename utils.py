from sklearn.metrics import confusion_matrix
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


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
    losses = 0
    for batch_idx, (data, target) in enumerate(train_loader):  # Iterate through the entire dataset to form an epoch.
        loss = train_iter(log_interval, model, device, optimizer, loss_func, data, target)  # Train for an iteration.
        losses += loss.item()
        if batch_idx % log_interval == 0:
            print('\nTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    return losses/len(train_loader)
            
def train_network(log_interval,model, device, train_loader, test_loader,optimizer, loss,thresholds=[0.5]*18,epochs=10):
    train_losses = []
    test_losses = []
    for epoch in tqdm(range(epochs)):
        train_loss = train_epoch(log_interval, model, device, train_loader, optimizer, epoch, loss)
        if test_loader:
            test_loss, _, _ = test(model,device,test_loader,loss_func=loss,thresholds=thresholds,target_available=True,verbose=True)
            test_losses.append(test_loss)
        
        train_losses.append(train_loss)
    if test_loader:
        return train_losses, test_losses
    else:
        return train_losses

def make_prediction(output, thresholds,device):
    thresholds = torch.as_tensor(thresholds).to(device)
    sigmoid_fun = torch.nn.Sigmoid()
    logits = sigmoid_fun(output)
    prediction = torch.where(logits >= thresholds, 1, 0)
    if prediction.sum() == 0:
        # If no logit is over threshold we predict the class with largest prob.
        prediction = torch.nn.functional.one_hot(logits.argmax(), num_classes=18).unsqueeze(0)
    return prediction

def test(model, device, test_loader, loss_func,thresholds=[0.5]*18,target_available=True,verbose=True):
    '''
    Testing the model on the test set and output predictions and possible target predictions
    '''
    model.eval()  # Switch the model to evaluation mode, which prevents the dropout behavior.
    all_preds = []
    all_targets = []
    if target_available:
        test_loss = 0
        tp = 0 # True positives
        fp = 0 # False positives
        fn = 0 # False negatives
        with torch.no_grad():  # Because this is testing and no optimization is required, the gradients are not needed.
            for data, target in (test_loader):  # Iterate through the entire test set.
                data, target = data.to(device), target.to(device)  # Move this batch of data to the specified device.
                all_targets.append(target.float())
                output = model(data)  # Forward the data through the model.
                test_loss += target.size(0)*loss_func(output, target.float()).item()  # Sum up batch loss
                pred = make_prediction(output,thresholds,device) # Get predictions in right format
                all_preds.append(pred)
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
        if verbose:
            print('\nTest set results: Average loss: {:.4f}, F1 Score: {:.2f}'.format(
                test_loss, F1))
        return test_loss, all_preds, all_targets
    else:
        for data, _ in tqdm(test_loader):  # Iterate through the entire test set.
            data = data.to(device)  # Move this batch of data to the specified device.
            output = model(data)  # Forward the data through the model.
            pred = make_prediction(output,thresholds,device)  # Get predictions in right format
            all_preds.append(pred)

        print('\nPredictions computed for test set.')
        return all_preds
    
def pos_weight(df_train,barplot=False,normalize=False):

    n = len(df_train)
    count=np.zeros(18)
    
    for i in range(n):
        labels = list(map(int,df_train.iloc[i,1].split(" ")))
        for j in range(18):
            if j < 11:
                count[j] = count[j]+labels.count(j+1)
            else:
                count[j] = count[j]+labels.count(j+2)
    
    print("Class counts: {}".format(torch.as_tensor(count)))
    total_samples = count.sum()
    # Compute weights (inverse frequency)
    class_weights = total_samples / count
    # Normalize weights (optional)
    if normalize:
        class_weights /= class_weights.max()
    print("Class weights: {}".format(torch.as_tensor(class_weights)))
    
    if barplot:
        names = np.array(np.arange(1,12),np.arange(13,20))
        plt.bar(names,count)
        plt.show()
    
    return torch.as_tensor(class_weights)