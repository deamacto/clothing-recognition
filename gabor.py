import numpy as np
from math import radians
from scipy import signal
import utils


def gabor(sigma, theta, Lambda, psi, gamma):
    sigma_x = sigma
    sigma_y = float(sigma) / gamma

    nstds = 3
    xmax = max(abs(nstds * sigma_x * np.cos(theta)), abs(nstds * sigma_y * np.sin(theta)))
    xmax = np.ceil(max(1, xmax))
    ymax = max(abs(nstds * sigma_x * np.sin(theta)), abs(nstds * sigma_y * np.cos(theta)))
    ymax = np.ceil(max(1, ymax))
    xmin = -xmax
    ymin = -ymax
    (y, x) = np.meshgrid(np.arange(-14, 15), np.arange(-14, 15))

    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)

    gb = np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) * np.cos(2 * np.pi / Lambda * x_theta + psi)
    return gb


def generate_gabor_filters():
    gabor_filters = []
    for i in range(0, 8):
        gabor_filters.append(gabor(sigma=5, theta=radians(i * 22.5), Lambda=15, psi=0, gamma=0.1))
    return gabor_filters


def apply_filters(image, filters):
    convolves = []

    for filter in filters:
        conv = signal.convolve2d(image, filter, mode='same')
        convolves.append(conv)

    sum = np.zeros(image.shape)

    for convolve in convolves:
        sum = np.add(sum, convolve)
    max = np.amax(sum)
    divisor = max // 254
    sum = (sum / 254).astype(int)

    return sum
