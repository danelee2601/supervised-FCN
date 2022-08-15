"""
It computes the FID score between two representation vectors.

reference: https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/
"""
# example of calculating the frechet inception distance
import torch
import numpy as np
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy.random import random
from scipy.linalg import sqrtm


# calculate frechet inception distance
def calculate_fid(z1, z2):
    """
    :param z1: representation after the last pooling layer (e.g., GAP)
    :param z2: representation after the last pooling layer (e.g., GAP)
    :return: FID score
    """
    # calculate mean and covariance statistics
    mu1, sigma1 = z1.mean(axis=0), cov(z1, rowvar=False)
    mu2, sigma2 = z2.mean(axis=0), cov(z2, rowvar=False)

    # calculate sum squared difference between means
    ssdiff = ((mu1 - mu2) ** 2.0).sum()

    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))

    # check and correct imaginary numbers from sqrt
    if iscomplexobj(covmean):
        covmean = covmean.real

    # calculate score
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


if __name__ == '__main__':
    # define two representation vectors
    z1 = torch.from_numpy(random(10 * 2048).reshape((10, 2048)))
    z2 = torch.from_numpy(random(10 * 2048).reshape((10, 2048)))

    # fid between z1 and z1
    fid = calculate_fid(z1, z1)
    print('FID (same): %.3f' % fid)

    # fid between z1 and z2
    fid = calculate_fid(z1, z2)
    print('FID (different): %.3f' % fid)
