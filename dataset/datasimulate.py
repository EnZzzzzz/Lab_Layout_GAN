import random

import numpy as np
import matplotlib.pyplot as plt
import torch
import visdom


def origin_center_alignment(batch_size):
    center = np.array([0, 0], dtype=np.float32)

    while True:
        dataset = []
        for i in range(batch_size):
            # size = np.random.randn(2).astype(np.float32)
            size = np.random.uniform(0.5, 1, (2)).astype(np.float32)
            lt = center - size
            rb = center + size
            dataset.append(np.concatenate((lt, rb), axis=0))
        dataset = np.array(dataset)
        yield dataset


def n_origin_center_alignment(n_point, batch_size):
    while True:
        center = np.zeros((batch_size, n_point, 2), dtype=np.float32)

        size = np.random.uniform(0, 1, (batch_size, n_point, 2)).astype(np.float32)
        lt = center - size
        rb = center + size
        points = np.concatenate((lt, rb), axis=2)
        yield points


def layout_data(batch_size, with_canvas=False):
    while True:
        canvas_size = [450, 450]
        prod_w_range, prod_h_range = [50, 400], [50, 225]
        text_w_range, text_h_range = [40, 350], [50, 150]
        padding = 50

        prod_w = np.random.uniform(prod_w_range[0], prod_w_range[1], (batch_size, 1)).astype(np.float32)
        prod_h = np.random.uniform(prod_h_range[0], prod_h_range[1], (batch_size, 1)).astype(np.float32)
        text_w = np.random.uniform(text_w_range[0], text_w_range[1], (batch_size, 1)).astype(np.float32)
        text_h = np.random.uniform(text_h_range[0], text_h_range[1], (batch_size, 1)).astype(np.float32)

        prod_x = 0.5 * (canvas_size[0] - prod_w)
        text_x = 0.5 * (canvas_size[0] - text_w)
        prod_y = 0.5 * (canvas_size[1] - prod_h - text_h - padding)
        text_y = prod_y + prod_h + padding

        prod_box = np.concatenate((prod_x, prod_y, prod_x + prod_w, prod_y + prod_h), axis=1)
        text_box = np.concatenate((text_x, text_y, text_x + text_w, text_y + text_h), axis=1)

        if with_canvas:
            canvas_box = np.array([0, 0, *canvas_size]).astype(np.float32)
            canvas_box = np.expand_dims(canvas_box, 0)
            canvas_box = canvas_box.repeat(batch_size, 0)
            data = np.stack((canvas_box, prod_box, text_box), 1)
        else:
            data = np.stack((prod_box, text_box), 1)
            # data = np.expand_dims(prod_box, 1)
        data = data / 450
        # print(data.shape)
        yield data


def visual_result(points, viz, title, n_fig=3):
    x = points[:, :, ::2]
    y = points[:, :, 1::2]

    x1, x2 = x[:, :, 0], x[:, :, 1]
    y1, y2 = y[:, :, 0], y[:, :, 1]
    box_x = np.stack((x1, x2, x2, x1, x1), axis=1)
    box_y = np.stack((y1, y1, y2, y2, y1), axis=1)

    canvas_x = np.array([0, 1, 1, 0, 0]).reshape((5, 1))
    canvas_y = np.array([0, 0, 1, 1, 0]).reshape((5, 1))

    over_idx = 0
    for i in range(n_fig):
        Y = np.concatenate((box_y[i], canvas_y), axis=1)
        X = np.concatenate((box_x[i], canvas_x), axis=1)

        viz.line(Y, X, win=f"{title}_{i}", opts={"title": f"{title}_{i}"})

        over_idx += 1


if __name__ == '__main__':
    # n_origin_center_alignment(2, 16)
    # data_iter = n_origin_center_alignment(1, 16)
    data_iter = layout_data(16)

    points = next(data_iter)

    import visdom

    viz = visdom.Visdom()

    visual_result(points, viz, "real", 16)
