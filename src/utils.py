import os
import numpy as np

def intra_mask(shapes, nans=False):
    mask = np.zeros(np.sum(shapes, axis=0))
    if nans:
        mask[:] = np.nan

    r, c = 0, 0
    for i, (rr, cc) in enumerate(shapes):
        if nans:
            values = np.zeros((rr,cc))
        else:
            values = np.ones((rr, cc))
        mask[r:r + rr, c:c + cc] = values
        r += rr
        c += cc

    return mask

def set_gpu(active=False):
    if not active:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = ""


def set_n_threads(n_threads):
    os.environ["OMP_NUM_THREADS"] = str(n_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n_threads)
    os.environ["MKL_NUM_THREADS"] = str(n_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(n_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n_threads)
