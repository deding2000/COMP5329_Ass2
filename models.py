from torch import nn
import torch
import torch.nn.functional as F

# Creata CNN
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()  # Compulsory operation.
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64,3,stride=1,padding=1)
        self.pool = nn.MaxPool2d(3,stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.dropout1 = nn.Dropout3d(1-0.9)
        self.dropout2 = nn.Dropout3d(1-0.75)
        self.dropout3 = nn.Dropout(1-0.5)
        self.fc1 = nn.Linear(14400, 1000)
        self.fc2 = nn.Linear(1000, 19)
    
    def forward(self, x):
        x = self.dropout1(x)
        x = self.pool(self.bn1(F.relu(self.conv1(x))))
        x = self.dropout2(x)
        x = self.pool(self.bn2(F.relu(self.conv2(x))))
        x = self.dropout2(x)
        x = self.pool(self.bn3(F.relu(self.conv3(x))))
        x = self.dropout2(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.dropout3(x)
        logits = self.fc2(x)
        return logits
    
# Creata CNN
class CNN_nodropout(nn.Module):
    def __init__(self):
        super(CNN_nodropout, self).__init__()  # Compulsory operation.
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128,3,stride=1,padding=1)
        self.pool = nn.MaxPool2d(3,stride=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        #self.dropout1 = nn.Dropout3d(1-0.9)
        #self.dropout2 = nn.Dropout3d(1-0.75)
        self.dropout3 = nn.Dropout(1-0.5)
        self.fc1 = nn.Linear(28800, 1575)
        self.fc2 = nn.Linear(1575, 19)
    
    def forward(self, x):
        #x = x.permute(1,0,2,3)
        #x = self.dropout1(x)
        x = self.pool(self.bn1(F.relu(self.conv1(x))))
        #x = self.dropout2(x)
        x = self.pool(self.bn2(F.relu(self.conv2(x))))
        #x = self.dropout2(x)
        x = self.pool(self.bn3(F.relu(self.conv3(x))))
        #x = self.dropout2(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.dropout3(x)
        logits = self.fc2(x)
        return logits
    
class CNN_no_batchnorm(nn.Module):
    def __init__(self):
        super(CNN_no_batchnorm, self).__init__()  # Compulsory operation.
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128,3,stride=1,padding=1)
        self.pool = nn.MaxPool2d(3,stride=2)
        #self.bn1 = nn.BatchNorm2d(32)
        #self.bn2 = nn.BatchNorm2d(64)
        #self.bn3 = nn.BatchNorm2d(128)
        self.dropout1 = nn.Dropout3d(1-0.9)
        self.dropout2 = nn.Dropout3d(1-0.75)
        self.dropout3 = nn.Dropout(1-0.5)
        self.fc1 = nn.Linear(28800, 1575)
        self.fc2 = nn.Linear(1575, 19)
    
    def forward(self, x):
        #x = x.permute(1,0,2,3)
        x = self.dropout1(x)
        x = self.pool((F.relu(self.conv1(x))))
        x = self.dropout2(x)
        x = self.pool((F.relu(self.conv2(x))))
        x = self.dropout2(x)
        x = self.pool((F.relu(self.conv3(x))))
        x = self.dropout2(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.dropout3(x)
        logits = self.fc2(x)
        return logits