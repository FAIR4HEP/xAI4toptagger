import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

class Net(nn.Module):
    def __init__(self, N=8, use_jet_pt=True, use_jet_mass=True, tau_x_1=False, hidden = [200, 200, 50, 50]):
        super().__init__()
        #num of features
        features = 3*(N-1)-1
        if tau_x_1:
            features = N-1
        if use_jet_mass:
            features+=1
        if use_jet_pt:
            features+=1

        if len(hidden) not in [3,4]:
            raise Exception("The model is currently setup to have 3 or 4 layers. Please adjust accordingly!")
        self.hidden = hidden
        self.dense1 = nn.Linear(features, hidden[0])
        self.dense2 = nn.Linear(hidden[0], hidden[1])
        self.dense3 = nn.Linear(hidden[1], hidden[2])
        if len(hidden) == 4:
            self.dense4 = nn.Linear(hidden[2], hidden[3])
            self.dense5 = nn.Linear(hidden[3], 2)
        else:
            self.dense4 = nn.Linear(hidden[2], 50) # This is a dummy layer, won't be used for anything really!
            self.dense5 = nn.Linear(hidden[2], 2)
        
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(.1)
        self.dropout2 = nn.Dropout(.2)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):

        x = self.dense1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        
        x = self.dense2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        
        x = self.dense3(x)
        x = self.relu(x)
        x = self.dropout1(x)
        
        if len(self.hidden) == 4:
            x = self.dense4(x)
            x = self.relu(x)
            x = self.dropout1(x)

        x = self.dense5(x)
        x = self.softmax(x)
        return x
    
    

if __name__ == "__main__":
    features=23
    model = Net().cuda()
    x = torch.rand(2, features).cuda()
    with torch.no_grad():
        y = model(x)
        print(y)
    summary(model, (1, 26))