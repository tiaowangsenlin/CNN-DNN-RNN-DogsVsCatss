
import torch.nn as nn

class myNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(200*200*3,512),
            nn.ReLU(),

            nn.Linear(512, 128),
            nn.ReLU(),

            nn.Linear(128, 2),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x=x.view(-1, 200 * 200 * 3)
        output = self.layers(x)
        return output