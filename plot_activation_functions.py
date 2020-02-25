import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

import numpy as np
from itertools import product

x = np.arange(-5, 5, 0.01)


def plot(ax, func, color, yaxis=(-1.4, 1.4)):
    # ax.ylim(yaxis)
    # ax.locator_params(nbins=5)
    # ax.xticks(fontsize = 14)
    # ax.yticks(fontsize = 14)
    # ax.axhline(lw=1, c='black')
    # ax.axvline(lw=1, c='black')
    # ax.grid(alpha=0.4, ls='-.')
    # ax.box(on=None)
    plt.plot(x, func(x), c=color, lw=3)


yaxis = (-5, 5)
xaxis = (-5, 5)

plt.figure()

plt.subplot(231, aspect='equal')
plt.ylim(yaxis)
plt.xlim(xaxis)
plt.axvline(x=0, color='k')
plt.axhline(y=0, color='k')
plt.xticks(np.arange(-5, 5.1, step=5))
plt.yticks(np.arange(-5, 5.1, step=5))
relu = np.vectorize(lambda x: x if x > 0 else 0, otypes=[np.float])
plt.plot(x, relu(x), c='r', lw=3)

plt.subplot(232, aspect='equal')
plt.ylim(yaxis)
plt.xlim(xaxis)
plt.axvline(x=0, color='k')
plt.axhline(y=0, color='k')
plt.xticks(np.arange(-5, 5.1, step=5))
plt.yticks(np.arange(-5, 5.1, step=5))
leaky_relu = np.vectorize(lambda x: max(0.1 * x, x), otypes=[np.float])
plt.plot(x, leaky_relu(x), c='r', lw=3)

plt.subplot(233, aspect='equal')
plt.ylim(yaxis)
plt.xlim(xaxis)
plt.axvline(x=0, color='k')
plt.axhline(y=0, color='k')
plt.xticks(np.arange(-5, 5.1, step=5))
plt.yticks(np.arange(-5, 5.1, step=5))
# \text{ELU}(x) = \max(0,x) + \min(0, \alpha * (\exp(x) - 1))
elu = np.vectorize(lambda x: x if x > 0 else 0.5 * (np.exp(x) - 1), otypes=[np.float])
plt.plot(x, elu(x), c='r', lw=3)

plt.subplot(234, aspect='equal')
plt.ylim(yaxis)
plt.xlim(xaxis)
plt.axvline(x=0, color='k')
plt.axhline(y=0, color='k')
plt.xticks(np.arange(-5, 5.1, step=5))
plt.yticks(np.arange(-5, 5.1, step=5))


def softplus(x_arg):
    return np.log(1 + np.exp(x_arg))


plt.plot(x, softplus(x), c='r', lw=3)

plt.subplot(235, aspect='equal')
plt.ylim(yaxis)
plt.xlim(xaxis)
plt.axvline(x=0, color='k')
plt.axhline(y=0, color='k')
plt.xticks(np.arange(-5, 5.1, step=5))
plt.yticks(np.arange(-5, 5.1, step=5))
# \text{CELU}(x) = \max(0,x) + \min(0, \alpha * (\exp(x/\alpha) - 1))
celu = np.vectorize(lambda x: x if x > 0 else 0.5 * (np.exp(x / 0.5) - 1), otypes=[np.float])
plt.plot(x, celu(x), c='r', lw=3)

plt.subplot(236, aspect='equal')
plt.ylim(yaxis)
plt.xlim(xaxis)
plt.axvline(x=0, color='k')
plt.axhline(y=0, color='k')
plt.xticks(np.arange(-5, 5.1, step=5))
plt.yticks(np.arange(-5, 5.1, step=5))
# \text{CELU}(x) = \max(0,x) + \min(0, \alpha * (\exp(x/\alpha) - 1))
prelu = np.vectorize(lambda x: max(0, x) + 0.2 * min(0, x), otypes=[np.float])
plt.plot(x, prelu(x), c='r', lw=3)

plt.show()
