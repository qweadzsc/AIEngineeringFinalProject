from torch import nn


class Predictor(nn.Module):

    def __init__(self, in_c, out_c, hid_d=1024):
        super().__init__()
        self.fc1 = nn.Linear(in_c, hid_d, bias=False)
        self.fc2 = nn.Linear(hid_d, out_c, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x))).sigmoid()
