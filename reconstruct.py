import time
import numpy as np
import skimage
import skimage.filters
import argparse
from utils import *
from algorithms import *


def eval_er(img_gt, img_rs, sup):
    er1 = er(img_gt, img_rs, sup)
    er2 = er(img_gt, np.rot90(img_rs, 2), sup)
    if er1 < er2:
        err = er1
    else:
        err = er2
        img_rs = np.rot90(img_rs, 2)
    return err, img_rs


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=str, default=None, help="Path to the input file")
    p.add_argument("--out", "-o", type=str, default=None, help="Path to put outputs")
    p.add_argument("--osr", type=int, default=5, help="Oversampling ratio")
    p.add_argument("--eta", type=int, default=0, help="Number of missing waves")
    p.add_argument("--beta", type=str, default=None, help="Poisson noise parameter")
    p.add_argument("--beta_hio", type=int, default=0.9, help="Beta parameter for HIO")
    p.add_argument("--algo", "-a", type=str, default="hio", help="Which algorithm to use")
    p.add_argument("--niter", "-n", type=int, default=100, help="Number of iterations")
    p.add_argument("--seed1", type=int, default=None, help="Random seed for Poisson noise")
    p.add_argument("--seed2", type=int, default=None, help="Random seed for initial guess")
    p.add_argument("--nexps", "-m", type=int, default=1, help="Number of independent reconstructions")
    p.add_argument("--lr_sigma", type=int, default=0, help="Sigma parameter for Gaussian blurring")
    p.add_argument("--lr_fom", type=str, default="E0", help="FOM for LR-guided reconstruction")
    p.add_argument("--hio_after", type=int, default="0",
                   help="Switch to HIO after x steps for LR-guided reconstruction")
    p.add_argument("--print_er_every", type=int, default=10, help="Print ER every x steps")
    p.add_argument("--print_ef_every", type=int, default=None, help="Print EF every x steps")
    p.add_argument("--recover_hq", action="store_true", help="Allow high-Q region with zero photons to be recovered")
    p.add_argument("--verbose", "-v", action="store_true", help="Be verbose")
    return p.parse_args()


def main():
    args = parse_args()
    img = np.load(args.input)
    if args.verbose:
        print("img:", img.shape)
    sup = np.pad(np.ones(img.shape), int((args.osr - 1) / 2 * img.shape[0]), mode="constant")
    img = oversample(img, args.osr)
    if args.verbose:
        print("img_pad:", img.shape)
    if args.algo == "lrg":
        img_lr = skimage.filters.gaussian(img, args.lr_sigma)
        img_lr = (np.sum(img)/np.sum(img_lr)) * img_lr
    fimg = simulate_diffraction(img, args.beta, seed=args.seed1)
    if args.verbose:
        print("fimg:", "min", np.min(fimg), "max", np.max(fimg))
    if args.eta > 0:
        w = 2 * args.eta * args.osr + 1
        if args.verbose:
            print("missing center: {}x{}".format(w, w))
        fimg, unk = remove_center(fimg, w)
        if args.recover_hq:
            unk = np.logical_or(unk, fimg == 0)
    else:
        unk = None
    for i in range(args.nexps):
        t0 = time.time()
        phase_seed = args.seed2 + i
        if args.nexps > 1:
            print("\nRandom seed: {}".format(phase_seed))
        if args.algo == "hio":
            img_rs, efs, ers = hio2d(fimg, args.niter, support=sup, beta=args.beta_hio, unknown=unk,
                                     er_every=args.print_er_every, ef_every=args.print_ef_every, seed=phase_seed,
                                     verbose=True, early_stop_cutoff=10, groundtruth=img)
        elif args.algo == "lrg":
            img_rs, efs, ers = lrg2d(fimg, args.niter, support=sup, beta=args.beta_hio, unknown=unk,
                                     er_every=args.print_er_every, ef_every=args.print_ef_every, seed=phase_seed,
                                     hio_after=args.hio_after, verbose=True, early_stop_cutoff=10, groundtruth=img,
                                     lr=img_lr, fom=args.lr_fom)
        else:
            raise NotImplementedError
        if args.verbose:
            print("time elapsed", time.time() - t0)
        err, img_rs = eval_er(img, img_rs, sup)
        print("ER={:.4f}".format(err))
        if args.out is not None:
            np.save("{}_{}".format(args.out, phase_seed), img_rs)
            if efs is not None:
                np.save("{}_{}_ef".format(args.out, phase_seed), efs)
            if ers is not None:
                np.save("{}_{}_er".format(args.out, phase_seed), ers)


if __name__ == "__main__":
    main()