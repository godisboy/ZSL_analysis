import torch.nn as nn

class AE_C(nn.Module):

    def __init__(self):
        super(AE_C, self).__init__()
        # encoding layers
        self.e1 = nn.Sequential(nn.Conv2d(3, 32, 4, 2, 1),
                                nn.BatchNorm2d(32),
                                nn.ReLU())
        self.e2 = nn.Sequential(nn.Conv2d(32, 32, 4, 2, 1),
                                nn.BatchNorm2d(32),
                                nn.ReLU())
        self.e3 = nn.Sequential(nn.Conv2d(32, 64, 4, 2, 1),
                                nn.BatchNorm2d(64),
                                nn.ReLU())
        self.e4 = nn.Sequential(nn.Conv2d(64, 64, 4, 2, 1),
                                nn.BatchNorm2d(64),
                                nn.ReLU())
        self.fc = nn.Sequential(nn.Linear(64 * 4 * 4, 512),
                                nn.ReLU())
        self.e = nn.Linear(512, 256)
        # decode layers
        self.d1 = nn.Linear(256, 512)
        self.d2 = nn.Sequential(nn.Linear(512, 64 * 4 * 4),
                                nn.ReLU())
        self.d3 = nn.Sequential(nn.ConvTranspose2d(64, 64, 4, 2, 1),
                                nn.BatchNorm2d(64),
                                nn.ReLU())
        self.d4 = nn.Sequential(nn.ConvTranspose2d(64, 32, 4, 2, 1),
                                nn.BatchNorm2d(32),
                                nn.ReLU())
        self.d5 = nn.Sequential(nn.ConvTranspose2d(32, 32, 4, 2, 1),
                                nn.BatchNorm2d(32),
                                nn.ReLU())
        self.d6 = nn.Sequential(nn.ConvTranspose2d(32, 3, 4, 2, 1),
                                nn.Tanh())

    def encode(self, x):
        h1 = self.e1(x)
        h2 = self.e2(h1)
        h3 = self.e3(h2)
        h4 = self.e4(h3)
        h4 = h4.view(h4.size(0), -1)
        h5 = self.fc(h4)
        return self.e(h5)

    def decode(self, z):
        h1 = self.d1(z)
        h2 = self.d2(h1)
        h2 = h2.view(h2.size(0), -1, 4, 4)
        h3 = self.d3(h2)
        h4 = self.d4(h3)
        h5 = self.d5(h4)
        return self.d6(h5)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)