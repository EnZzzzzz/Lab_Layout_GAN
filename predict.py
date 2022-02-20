import torch

from dataset.datasimulate import visual_result
from model import Generator
from model import Discriminator
import visdom

viz = visdom.Visdom()

batch_size = 1
noise_dim = 4
n_box = 2
feature_dim = 4
lr = 1e-4
epoch_num = 40000
h_dim = 200

G = Generator(noise_dim, feature_dim, n_box, h_dim)
G.load_state_dict(torch.load("ckpl/generator_40000.pth"))

G.cuda()
# z = torch.randn(batch_size, n_box, noise_dim).cuda()
z = torch.zeros((16, 2, 4)).cuda()
print(z)

points = G(z).detach().cpu().numpy()
visual_result(points, viz, "prediction", 16)
