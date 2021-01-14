import numpy as np
from numpy import angle, absolute
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from metrics import *


def hio2d(x, niters, support, init_phase=None, unknown=None, beta=0.9, ef_every=None, er_every=None,
          groundtruth=None, seed=None, verbose=False, early_stop_cutoff=-1, squared=True):
    if init_phase is None:
        rg = np.random.default_rng(seed)
        init_phase = np.angle(fft2(rg.random(x.shape)))
    else:
        pass
    if unknown is None:
        unknown = np.zeros(x.shape)
    else:
        pass
    if ef_every:
        pass
    else:
        ef_every = 0
    efs = list()
    if er_every:
        pass
    else:
        er_every = 0
    ers = list()
    mag = np.absolute(x)
    x = mag * np.exp(1j*init_phase)
    x_prev = ifft2(x).real
    ef_prev = -1
    conv_count = 0
    for i in range(niters):
        x = ifft2(x).real
        mask = np.logical_or((support < 1), (x < 0))
        # print("mask", np.sum(1*mask)/mask.size)
        x[mask] = x_prev[mask] - beta * x[mask]
        x_prev = x
        if er_every > 0:
            if (i+1) % er_every == 0 or (i+1) < er_every:
                er1 = er(groundtruth, x, support)
                er2 = er(groundtruth, np.rot90(x, 2), support)
                if er1 < er2:
                    ers.append((i+1, er1))
                    if verbose:
                        print("iter {:5d} ER: {:.4e}".format(i+1, er1))
                else:
                    ers.append((i+1, er2))
                    if verbose:
                        print("iter {:5d} ER: {:.4e} (flipped)".format(i+1, er2))
        x = np.fft.fft2(x)
        ef1 = ef(mag, x, unknown, squared=squared)
        if np.abs((ef1 - ef_prev) / ef_prev) < 1e-4:
            if verbose:
                print("Delta", np.abs(ef1 - ef_prev) / float(ef_prev))
            conv_count += 1
        else:
            conv_count = 0
        if conv_count >= early_stop_cutoff > 0:
            if verbose:
                print("early stop after {} steps".format(i))
            break
        ef_prev = ef1
        if ef_every > 0:
            if (i+1) % ef_every == 0 or (i+1) < ef_every:
                efs.append((i+1, ef1))
                if verbose:
                    # print("fimg max: {:.4e}".format(np.max(np.square(np.abs(x)).real)) )
                    print("iter {:5d} EF: {:.4e}".format(i+1, ef1))
        x[unknown < 1] = mag[unknown < 1] * np.exp(1j*np.angle(x[unknown < 1]))
    x = np.fft.ifft2(x).real
    efs = np.array(efs)
    ers = np.array(ers)
    return x, efs, ers


def lrg2d(x, niters, support, lr, init_phase=None, unknown=None, beta=0.9, gamma0=1.0, ef_every=None, er_every=None,
          groundtruth=None, seed=None, fom="E0", hio_after=0, normalized_to_all=True, squared=True, rms=True,
          verbose=False, early_stop_cutoff=-1):
    if init_phase is None:
        rg = np.random.default_rng(seed)
        init_phase = np.angle(fft2(rg.random(x.shape)))
    else:
        pass
    if unknown is None:
        unknown = np.zeros(x.shape)
    else:
        pass
    if ef_every:
        pass
    else:
        ef_every = 0
    efs = list()
    if er_every:
        pass
    else:
        er_every = 0
    ers = list()
    if fom == "E0":
        print("FOM: E0")
        use_e0 = True
    else:
        print("FOM: EF")
        use_e0 = False
    mag = np.absolute(x)
    x = mag * np.exp(1j*init_phase)
    x_prev = ifft2(x).real
    ef_prev = -1
    conv_count = 0
    for i in range(niters):
        x = ifft2(x).real
        mask = np.logical_or((support < 1), (x < 0))
        # print("mask", np.sum(1 * mask) / mask.size)
        x[mask] = x_prev[mask] - beta * x[mask]
        ef1 = ef(mag, fft2(x), unknown, squared=squared)
        if np.abs(ef1-ef_prev)/float(ef_prev) < 1e-4:
            conv_count += 1
        else:
            conv_count = 0
        if conv_count >= early_stop_cutoff > 0 and i > (hio_after + early_stop_cutoff):
            print("early stop after {} steps".format(i))
            x = np.fft.fft2(x)
            break
        ef_prev = ef1
        if ef_every > 0:
            if (i+1) % ef_every == 0:
                efs.append((i+1, ef1))
                if verbose:
                    print("iter {:5d} EF: {:.4e}".format(i+1, ef1))
        if use_e0:
            gamma = min(e0(x, support, normalized_to_all=normalized_to_all, rms=rms), 1)
        else:
            gamma = ef1
        # print(gamma)
        if hio_after == 0 or i < hio_after:
            x = x + gamma0 * gamma * (lr-x)
        elif i >= hio_after:
            pass
            # x = x + np.exp(lam*(hio_after-i)) * gamma0 * gamma * (lr-x)
        else:
            pass
        x_prev = x
        if er_every > 0:
            if (i+1) % er_every == 0 or (i+1) < er_every:
                er1 = er(groundtruth, x, support)
                er2 = er(groundtruth, np.rot90(x, 2), support)
                if er1 < er2:
                    ers.append((i+1, er1))
                    if verbose:
                        print("iter {:5d} ER: {:.4e} FOM: {:.4e}".format(i+1, er1, gamma))
                else:
                    ers.append((i+1, er2))
                    if verbose:
                        print("iter {:5d} ER: {:.4e} FOM: {:.4e}(flipped)".format(i+1, er2, gamma))
        x = np.fft.fft2(x)
        x[unknown < 1] = mag[unknown < 1] * np.exp(1j*np.angle(x[unknown < 1]))
    x = np.fft.ifft2(x).real
    efs = np.array(efs)
    ers = np.array(ers)
    return x, efs, ers
