import numpy as np


def er(gt_image, x_image, support):
    err = np.sum(np.abs(gt_image[support > 0] - x_image[support > 0]))
    err = err / np.sum(np.abs(gt_image[support > 0]))
    return err


def ef(f_obs, f_x, support, squared=False):
    if squared:
        err = np.sum(np.square(np.abs(np.abs(f_obs[support < 1]) - np.abs(f_x[support < 1]))))
        err = err / np.sum(np.square(np.abs(f_obs[support < 1])))
    else:
        err = np.sum(np.abs(np.abs(f_obs[support < 1]) - np.abs(f_x[support < 1])))
        err = err / np.sum(np.abs(f_obs[support < 1]))
    return err


def e0(f_x, support, normalized_to_all=False, rms=True):
    if rms:
        err = np.sum(np.square(np.abs(f_x[support < 1])))
        if normalized_to_all:
            err = np.sqrt(err) / np.sqrt(np.sum(np.square(np.abs(f_x))))
        else:
            err = np.sqrt(err) / np.sqrt(np.sum(np.square(np.abs(f_x[support > 0]))))
    else:
        err = np.sum(np.abs(f_x[support < 1]))
        if normalized_to_all:
            err = err / np.sum((np.abs(f_x)))
        else:
            err = err / np.sum((np.abs(f_x[support > 0])))
    return err
