from torch import nn


class Generator(nn.Module):

    def __init__(self, noise_dim, feature_dim, n_box, h_dim=400):
        super(Generator, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(noise_dim, h_dim),
            nn.ReLU(True),

            nn.Dropout(),
            nn.Linear(h_dim, h_dim),
            # nn.BatchNorm1d(n_box),
            nn.ReLU(True),

            nn.Dropout(),
            nn.Linear(h_dim, h_dim),
            # nn.BatchNorm1d(n_box),
            nn.ReLU(True),

            nn.Dropout(),
            nn.Linear(h_dim, feature_dim),
        )

    def forward(self, z):
        output = self.net(z)
        return output
