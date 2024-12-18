import numpy as np
import MCMCIterators.samplers as samplers
import matplotlib.pyplot as plt

def gauss_logpdf(x):
    cov = np.array([[1.0, 0.2], 
                    [0.2, 2.0]])
    mean = np.array([1.0, 2.0])

    diff = x - mean
    return -0.5 * np.dot(diff,  np.linalg.solve(cov, diff))


init_sample = np.array([0.0, 0.0])
init_cov = np.array([[1.0, 0.0], 
                     [0.0, 1.0]])

sampler = samplers.DelayedRejectionAdaptiveMetropolis(
    gauss_logpdf, init_sample, init_cov,
    adapt_start=10,
    eps=1e-6, sd=None,
    interval=1, level_scale=1e-1,
)

max_samples = 10000
x, y = [init_sample[0]], [init_sample[1]]
accepted = 0
for i, (sample, logpdf, accepted_bool) in enumerate(sampler):
    print(f"Sample: {sample}")
    print(f"\t Logpdf: {logpdf}")
    print(f"\t Accepted? -> {accepted_bool}")
    print("\n")
    x.append(sample[0])
    y.append(sample[1])
    if accepted_bool: 
        accepted += 1
    
    if i > max_samples:
        break

print(f"Acceptance rate: {accepted / max_samples * 100}%")

min, max = -2, 6
N = 200
coords = np.linspace(min, max, N)
X, Y = np.meshgrid(coords, coords)
logpdf = lambda x, y: gauss_logpdf(np.array([x,y]))

pdf = np.zeros((N, N))

for i in range(N):
    for j in range(N):
        pdf[j, i] = np.exp(gauss_logpdf(np.array([X[j,i], Y[j,i]])))


plt.scatter(x, y, alpha = 0.01)
plt.contour(X, Y, pdf)
plt.show()
