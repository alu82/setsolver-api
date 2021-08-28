import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import torchvision.transforms as transforms

class SetCardClassifier(nn.Module):
    def __init__(self):
        super(SetCardClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 5, padding=1)
        self.conv5 = nn.Conv2d(128, 256, 5, padding=1)
        self.pool = nn.MaxPool2d(2, 2) 
        self.fc1 = nn.Linear(3*6*256, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 81)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # start: 160x250x3, after: 80x125x16
        x = self.pool(F.relu(self.conv2(x))) # after: 40x62x32
        x = self.pool(F.relu(self.conv3(x))) # after: 20x31x64
        x = self.pool(F.relu(self.conv4(x))) # after: 9x14x128
        x = self.pool(F.relu(self.conv5(x))) # after: 3x6x256
        x = x.view(-1, 3*6*256) # flatten image input
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = F.log_softmax(self.fc3(x), dim=1)
        return x
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = f"{os.path.dirname(__file__)}/model_01.pt"
state_dict = torch.load(model_path, map_location=device)

classifier = SetCardClassifier()
classifier.load_state_dict(state_dict)
classifier.eval()

def classify_card(image):
    mean = [0.6, 0.6, 0.6]
    std = [0.2, 0.2, 0.2]
    input_transforms = transforms.Compose([
        transforms.Resize((250,160)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])

    timg = input_transforms(image)
    timg = torch.unsqueeze(timg, 0)
    return get_prediction(timg)

def get_prediction(input_tensor):
    log_probabilities = classifier.forward(input_tensor)
    probabilities = torch.exp(log_probabilities)
    top_prob, top_class = probabilities.topk(1, dim=1)
    return top_prob.item(), top_class.item()