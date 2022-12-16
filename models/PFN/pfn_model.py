import torch
import torch.nn as nn
from torchinfo import summary

class ParticleFlowNetwork(nn.Module):
    r"""Parameters
    ----------
    input_dims : int
        Input feature dimensions.
    num_classes : int
        Number of output classes.
    """

    def __init__(self, input_dims=3, num_classes=2,
                 Phi_sizes=(100, 100, 256),
                 F_sizes=(100, 100, 100),
                 for_inference=True,
                 **kwargs):

        super(ParticleFlowNetwork, self).__init__(**kwargs)
        # per-particle functions
        print(Phi_sizes, F_sizes)
        phi_layers = []
        for i in range(len(Phi_sizes)):
            phi_layers.append(nn.Sequential(
                nn.Linear(input_dims if i == 0 else Phi_sizes[i - 1], Phi_sizes[i]),
                nn.ReLU())
            )
        self.phi = nn.Sequential(*phi_layers)
        # global functions
        f_layers = []
        for i in range(len(F_sizes)):
            f_layers.append(nn.Sequential(
                nn.Linear(Phi_sizes[-1] if i == 0 else F_sizes[i - 1], F_sizes[i]),
                nn.ReLU())
            )
        f_layers.append(nn.Linear(F_sizes[-1], num_classes))
        if for_inference:
            f_layers.append(nn.Softmax(dim=1))
        self.fc = nn.Sequential(*f_layers)

    def forward(self, features, mask):
        # x: the feature vector initally read from the data structure, in dimension (N, C, P)
        #features = features.permute(0,2,1)
        features = torch.flatten(features, start_dim=0, end_dim=1)
        x = self.phi(features)
        x = torch.stack(torch.split(x.permute(1, 0), 200, dim=1), 0)
        if mask is not None:
            x = x * mask.bool().float()
        x = x.sum(-1)
        return self.fc(x)


if __name__ == "__main__":
    features=3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ParticleFlowNetwork(input_dims=features).to(device)
    x = torch.rand(1, 200, features).to(device)
    with torch.no_grad():
        y = model(x, None)
        print(y)
    summary(model, ((1, 200, 3), (1, 1, 200)))
