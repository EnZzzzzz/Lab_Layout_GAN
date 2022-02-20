import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import torch
import visdom
from torch import optim

from dataset.datasimulate import origin_center_alignment, visual_result, n_origin_center_alignment, layout_data
from model import Generator, Discriminator
from util import weights_init, gradient_penalty


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    viz = visdom.Visdom()
    torch.manual_seed(23)
    np.random.seed(23)

    batch_size = 256
    noise_dim = 4
    n_box = 2
    feature_dim = 4
    lr = 1e-4
    epoch_num = 40000
    h_dim = 200

    G = Generator(noise_dim, feature_dim, n_box, h_dim).to(device)
    D = Discriminator(feature_dim, n_box, h_dim).to(device)
    G.apply(weights_init)
    D.apply(weights_init)

    optim_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.9))
    optim_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.9))

    data_iter = layout_data(batch_size)
    print('batch:', next(data_iter).shape)

    for epoch in range(epoch_num):
        for _ in range(5):
            x = next(data_iter)
            xr = torch.from_numpy(x).to(device)

            predr = (D(xr))
            lossr = -(predr.mean())

            # [b, 2]
            z = torch.randn(batch_size, n_box, noise_dim).cuda()
            # stop gradient on G
            # [b, 2]
            xf = G(z).detach()
            # [b]
            predf = (D(xf))
            # min predf
            lossf = (predf.mean())

            # gradient penalty
            gp = gradient_penalty(D, xr, xf, batch_size)

            loss_D = lossr + lossf + gp
            optim_D.zero_grad()
            loss_D.backward()
            # for p in D.parameters():
            #     print(p.grad.norm())
            optim_D.step()

        # 2. train Generator
        z = torch.randn(batch_size, n_box, noise_dim).cuda()
        xf = G(z)
        predf = (D(xf))
        # max predf
        loss_G = - (predf.mean())
        optim_G.zero_grad()
        loss_G.backward()
        optim_G.step()

        if epoch % 20 == 0:
            viz.line([[loss_D.item(), loss_G.item(), gp.item()]],
                     [[epoch, epoch, epoch]],
                     win='loss',
                     update='append',
                     opts={"title": "loss", "legend": ["loss d", "loss g", "gp"]}
                     )
            visual_result(xf.detach().cpu().numpy(), viz, "pred", 4)
            visual_result(xr.detach().cpu().numpy(), viz, "real", 2)
            # generate_image(D, G, xr, epoch)
            print(loss_D.item(), loss_G.item(), gp.item())

    torch.save(G.state_dict(), f"ckpl/generator_{epoch_num}.pth")
    torch.save(D.state_dict(), f"ckpl/discriminator_{epoch_num}.pth")


if __name__ == '__main__':
    main()
