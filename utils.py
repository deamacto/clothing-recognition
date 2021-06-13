from math import sqrt
import numpy as np
from matplotlib import pyplot as plt


def squarify(vec, side_length=None):
    if side_length is None:
        side_length = int(sqrt(len(vec)))
    return np.reshape(vec, (side_length, side_length))


it = 1


def plot_as_img(arr_img):
    global it
    plt.figure(it)
    it += 1
    plt.imshow(arr_img)
