from model import keypoint_model, orientation_model
from utils import post_process_orient, Transformer, post_process_kp
from utils import Transformer

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import imageio

kp_model = keypoint_model()
kp_model.load_weights("./keypoint_network_planes.h5")

image_files = ["./output/"+i for i in os.listdir("./output/") if i.endswith("png")]

def draw_circle(rgb, u, v, col, r):
    """Draws a simple anti-aliasing circle in-place.

    Args:
    rgb: Input image to be modified.
    u: Horizontal coordinate.
    v: Vertical coordinate.
    col: Color.
    r: Radius.
    """

    ir = int(math.ceil(r))
    for i in range(-ir-1, ir+2):
        for j in range(-ir-1, ir+2):
            nu = int(round(u + i))
            nv = int(round(v + j))
            if nu < 0 or nu >= rgb.shape[1] or nv < 0 or nv >= rgb.shape[0]:
                continue

            du = abs(nu - u)
            dv = abs(nv - v)

            # need sqrt to keep scale
            t = math.sqrt(du * du + dv * dv) - math.sqrt(r * r)
            if t < 0:
                rgb[nv, nu, :] = col
            else:
                t = 1 - t
                if t > 0:
                    # t = t ** 0.3
                    rgb[nv, nu, :] = col * t + rgb[nv, nu, :] * (1-t)


def draw_ndc_points(rgb, xy, cols):
    """Draws keypoints onto an input image.

    Args:
    rgb: Input image to be modified.
    xy: [n x 2] matrix of 2D locations.
    cols: A list of colors for the keypoints.
    """

    vh, vw = rgb.shape[0], rgb.shape[1]

    for j in range(len(cols)):
        x, y = xy[j, :2]
        x = (min(max(x, -1), 1) * vw / 2 + vw / 2) - 0.5
        y = vh - 0.5 - (min(max(y, -1), 1) * vh / 2 + vh / 2)

        x = int(round(x))
        y = int(round(y))
        if x < 0 or y < 0 or x >= vw or y >= vh:
            continue

        rad = 1.5
        rad *= rgb.shape[0] / 128.0
        draw_circle(rgb, x, y, np.array([0.0, 0.0, 0.0, 1.0]), rad * 1.5)
        draw_circle(rgb, x, y, cols[j], rad)

cols = plt.cm.get_cmap("rainbow")(
      np.linspace(0, 1.0, 10))[:, :4]

for f in image_files:
    img = imageio.imread(f).astype(float) / 255
    if img.shape[2] == 3:
        img = np.concatenate((img, np.ones_like(img[:, :, :1])), axis=2)
    prob, z = kp_model(np.expand_dims(img, 0))
    uv, z = post_process_kp(prob, z)
    uvz = np.concatenate((uv, np.expand_dims(z, axis=-1)), axis=2)

    draw_ndc_points(img, np.reshape(uvz, (10,3)), cols)
    imageio.imsave("results/" + f.split("/")[-1], img)