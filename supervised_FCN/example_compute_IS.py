"""
It computes the IS (Inception Score) between two representation vectors.
NB! though we're using the term IS, FCN (Fully Convolutional Network) [1] is used instead of Inception.
The use of FCN for IS is from [2].

reference: https://machinelearningmastery.com/how-to-implement-the-inception-score-from-scratch-for-evaluating-generated-images/

[1] Wang, Zhiguang, Weizhong Yan, and Tim Oates. "Time series classification from scratch with deep neural networks: A strong baseline." 2017 International joint conference on neural networks (IJCNN). IEEE, 2017.
[2] Smith, Kaleb E., and Anthony O. Smith. "Conditional GAN for timeseries generation." arXiv preprint arXiv:2006.16477 (2020)
"""
# calculate inception score in numpy
import numpy as np
from numpy import asarray
from numpy import expand_dims
from numpy import log
from numpy import mean
from numpy import exp


# calculate the inception score for p(y|x)
def calculate_inception_score(P_yx, n_split: int = 10, shuffle: bool = True, eps: float = 1e-16):
    """
    P_yx: (batch_size dim)
    """
    if shuffle:
        np.random.shuffle(P_yx)  # in-place

    scores = list()
    n_part = int(np.floor(P_yx.shape[0] / n_split))
    for i in range(n_split):
        # retrieve p(y|x)
        ix_start, ix_end = i * n_part, i * n_part + n_part
        p_yx = P_yx[ix_start:ix_end]

        # calculate p(y)
        p_y = expand_dims(p_yx.mean(axis=0), 0)

        # calculate KL divergence using log probabilities
        kl_d = p_yx * (log(p_yx + eps) - log(p_y + eps))

        # sum over classes
        sum_kl_d = kl_d.sum(axis=1)

        # average over images
        avg_kl_d = mean(sum_kl_d)

        # undo the log
        is_score = exp(avg_kl_d)

        # store
        scores.append(is_score)

    # average across images
    is_avg, is_std = np.mean(scores), np.std(scores)
    return is_avg, is_std


if __name__ == '__main__':
    # assume that we have 50 samples and 3 classes.
    # generate synthetic `p(y|x)`
    rand = np.random.rand(50, 3)
    p_yx = rand / np.sum(rand, axis=1, keepdims=True)  # p(y|x)

    # calculate inception score
    is_avg, is_std = calculate_inception_score(p_yx)
    print(f"""
    IS (mean): {is_avg}
    IS (std): {is_std}
    """)

