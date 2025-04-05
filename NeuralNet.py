import torch.nn as nn
import torch

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.out = nn.Sequential(
            nn.Linear(28*28, 10),
            nn.LogSigmoid(),
            nn.Linear(10, 10),
            nn.LogSigmoid(),
            nn.Linear(10, 10),
            nn.LogSigmoid()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.out(x)
        return logits
    
# t-SNE
# if we put all the outputs of different minimums we think we've reached and are able to cluster them we can show theyre different minimums as they would be different distributions.
