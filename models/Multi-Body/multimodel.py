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
        layers = [features] + hidden + [2]
        nn_layers = []
        for i in range(1,len(layers)):
            drate = 0.2 if i < len(layers)/2 else 0.1
            nn_layers.append(nn.Sequential(
                nn.Linear(layers[i - 1], layers[i]),
                nn.ReLU() if i < len(layers) - 1 else nn.Identity(),
                nn.Dropout(drate) if i < len(layers) - 1 else nn.Identity())
            )
        self._nn = nn.Sequential(*nn_layers)
        self.softmax = nn.Softmax(dim=1)
        self.features = features
        
    def forward(self, x):
        return self.softmax(self._nn.forward(x))
    
    

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net().to(device)
    features = model.features
    x = torch.rand(2, features).to(device)
    with torch.no_grad():
        y = model(x)
        print(y)
    summary(model)
