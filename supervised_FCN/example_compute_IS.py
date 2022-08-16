"""
It computes the IS (Inception Score) between two representation vectors.
NB! though we're using the term IS, FCN (Fully Convolutional Network) [1] is used instead of Inception.
The use of FCN for IS is from [2].

reference: https://machinelearningmastery.com/how-to-implement-the-inception-score-from-scratch-for-evaluating-generated-images/

[1] Wang, Zhiguang, Weizhong Yan, and Tim Oates. "Time series classification from scratch with deep neural networks: A strong baseline." 2017 International joint conference on neural networks (IJCNN). IEEE, 2017.
[2] Smith, Kaleb E., and Anthony O. Smith. "Conditional GAN for timeseries generation." arXiv preprint arXiv:2006.16477 (2020)
"""
# calculate inception score in numpy
from numpy import asarray
from numpy import expand_dims
from numpy import log
from numpy import mean
from numpy import exp


# calculate the inception score for p(y|x)
def calculate_inception_score(p_yx, eps=1e-16):
    # calculate p(y)
    p_y = expand_dims(p_yx.mean(axis=0), 0)

    # kl divergence for each image
    kl_d = p_yx * (log(p_yx + eps) - log(p_y + eps))  # KL divergence = p(y|x) * (log(p(y|x)) – log(p(y)))

    # sum over classes
    sum_kl_d = kl_d.sum(axis=1)

    # average over images
    avg_kl_d = mean(sum_kl_d)

    # undo the logs
    is_score = exp(avg_kl_d)
    return is_score


if __name__ == '__main__':
    # we can imagine the case of three classes of image and a perfect confident prediction for each class for three images.
    # conditional probabilities for high quality images
    p_yx = asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])  # p(y|x)

    # score
    score = calculate_inception_score(p_yx)
    print(score)

    # we can also try the worst case.
    # conditional probabilities for low quality images
    p_yx = asarray([[0.33, 0.33, 0.33], [0.33, 0.33, 0.33], [0.33, 0.33, 0.33]])
    score = calculate_inception_score(p_yx)
    print(score)
