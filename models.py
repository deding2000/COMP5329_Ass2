from torch import nn
import torch
import torch.nn.functional as F

# Creata CNN
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()  # Compulsory operation.
        self.conv1 = nn.Conv3d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=3)
        self.conv2 = nn.Conv3d(32, 64, 5, stride=1, padding=2)
        self.pool = nn.MaxPool3d(2,2)
        self.dropout1 = nn.Dropout3d(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(18513, 128)
        self.fc2 = nn.Linear(128, 19)
    
    def forward(self, x):
        x = x.permute(1,0,2,3)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        return logits