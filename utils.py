import numpy as np
from numpy import angle, absolute
from numpy.fft import fft2, ifft2, fftshift, ifftshift


def remove_center(fimg, w):
    fimg = fftshift(fimg)
    unk = np.ones((w, w))
    pad_size = fimg.shape[0]-w
    if pad_size % 2 == 0:
        unk = np.pad(unk, int((fimg.shape[0]-w)/2), mode="constant")
    else:
        p = int((fimg.shape[0] - w) / 2)
        unk = np.pad(unk, ((p+1, p), (p+1, p)))
    fimg[unk > 0] = 0
    fimg = ifftshift(fimg)
    unk = ifftshift(unk)
    return fimg, unk


def oversample(img, osr):
    return np.pad(img, int((osr-1)/2*img.shape[0]), mode="constant")


def simulate_diffraction(img, beta=None, seed=None):
    rg = np.random.default_rng(seed)
    fimg = np.square(np.absolute(np.fft.fft2(img))).real
    nmax = np.max(fimg)
    if beta is None:
        pass
    else:
        beta = np.float(beta)
        if beta > 0:
            fimg = (nmax/beta) * rg.poisson((beta/nmax)*fimg)
    fimg = np.sqrt(fimg)
    return fimg

