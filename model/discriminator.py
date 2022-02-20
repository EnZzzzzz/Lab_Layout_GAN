from torch import nn


class Discriminator(nn.Module):

    def __init__(self, feature_dim, n_box, h_dim=400):
        super(Discriminator, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(feature_dim, h_dim),
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
            nn.Linear(h_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        output = self.net(x)
        return output.view(-1)
