from matplotlib import pyplot as plt
import numpy as np
import scipy
import scipy.stats

rng = np.random.default_rng(0)
n = 20
x = rng.normal(size=n)
e = rng.normal(size=n) / 5
y = .5 * x + .1 * x**2 + e
reg = scipy.stats.linregress(x, y)

plt.scatter(x, y)
grid = np.sort(x)
plt.plot(grid, reg.intercept + reg.slope * grid, color="k")
plt.gca().axis("off")
plt.gcf().savefig("/tmp/fig.png", bbox_inches="tight")
