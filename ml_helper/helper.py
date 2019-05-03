from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as nlg
import numpy.random as npr
from scipy.special import expit


def gen_example_samples(seed=1337):

    npr.seed(seed)
    N = 30

    X = npr.uniform(low=-5, high=5, size=N)
    Y = 5 * X + 10 + 0.1 * X**2

    noisy = npr.binomial(1, 0.05, size=N).astype(bool)
    Y[noisy] = 0

    return X, Y


def gen_planar_samples(
        *, complexity=10, noisiness=0.33, num=256, xylim=(-5, 5), seed=None
):
    '''
    Generates random 2D features and labels based on plane waves.
    '''

    if seed:
        npr.seed(seed)

    # hack to work around late binding
    def xsin(amp, k, phi, xy):
        return amp * np.sin(k @ xy + phi)

    funcs = []
    for i in range(complexity):
        # wavevector, factor out scale
        k = npr.uniform(i / 2, i + 1, size=(2, )) / xylim[1]
        k *= (2 * (npr.random(size=(2, )) - 0.5))
        # phase
        phi = npr.uniform(0, 2 * np.pi)
        # amplitude
        amp = xylim[1] / nlg.norm(k)

        funcs.append(partial(xsin, amp, k, phi))

    def amplitude(xys):
        def amp_row(xy):
            return sum(f(xy) for f in funcs)

        amp = np.apply_along_axis(amp_row, 1, xys)
        amp -= amp.mean()
        amp /= np.std(amp)
        amp = expit(amp)

        return amp

    xys = npr.uniform(xylim[0], xylim[1], size=(num, 2))
    amp = amplitude(xys)

    # will work without size, but will be WRONG
    ythresh = noisiness * npr.random(size=xys.shape[:1])

    y = np.zeros(len(xys))
    y[amp > 0.5 * (1 - noisiness) + ythresh] = 1

    return xys, y, amplitude


def plot_decision_surface(
        fpred,
        xlim=(-5, 5),
        ylim=(-5, 5),
        ax=None,
        with_data=None,
        size=(14, 8),
        with_true_surface=None,
        binary=False,
        cutoff=0.5,
        title=None,
        xlabel=None,
        ylabel=None,
):

    if ax and with_true_surface:
        raise ValueError('Cannot plot two surfaces with single passed axes!')

    xmn, xmx = xlim
    ymn, ymx = ylim

    XX, YY = np.meshgrid(np.arange(xmn, xmx, 0.05), np.arange(ymn, ymx, 0.05))

    plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

    Z = fpred(np.c_[XX.ravel(), YY.ravel()])

    if binary:
        which = np.where(Z < cutoff)[0]
        Z[:] = 1
        Z[which] = 0

    # specialcase two classes, dirty but worx
    if len(Z.shape) == 2 and Z.shape[1] == 2:
        Z = Z[:, 1]

    Z = Z.reshape(XX.shape)

    if not ax:
        if with_true_surface:
            ax = plt.subplot(1, 2, 1)
            ax_true = plt.subplot(1, 2, 2)
        else:
            ax = plt.gca()

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        # set size only if we own the axes
        plt.gcf().set_size_inches(size)

    ax.contourf(
        XX,
        YY,
        Z,
        cmap=plt.cm.RdYlBu,
        vmin=0,
        vmax=1,
    )

    if with_data:
        x, y = with_data
        plot_red_blue(x, y, ax=ax)

    if with_true_surface:
        plot_decision_surface(
            with_true_surface, xlim=xlim, ylim=ylim, ax=ax_true
        )

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    plt.tight_layout()


def plot_red_blue(x, y, ax=None):
    if not ax:
        plt.figure()
        ax = plt.gca()

    xred = x[y == 0]
    xblue = x[y == 1]
    ax.scatter(xred[:, 0], xred[:, 1], color='red', s=5)
    ax.scatter(xblue[:, 0], xblue[:, 1], color='blue', s=5)
