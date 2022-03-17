from collections import defaultdict
import numpy as np

# Some helper functions that I wrote
# NOTE: Not sure this gaussian kernel is actually correct
def gaussian_kernel(x, y, bandwidth=1):
    return np.exp(np.linalg.norm((x-y), ord=2) ** 2 / (-bandwidth ** 2))


# The code below this line was written by Arturs Backurs, Piotr Indyk and Tal Wagner.
"""
MIT License

Copyright (c) 2019 talwagner

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""
### Space and time efficient kernel density estimation in high dimensions
### Code for NeurIPS 2019 paper by Arturs Backurs, Piotr Indyk and Tal Wagner

### KDE

def laplacian_kernel(x, y, bandwidth=1):
    "Laplacian kernel"
    return np.exp(-1. * np.linalg.norm((x-y), ord=1) / bandwidth)

def kde(query, dataset, bandwidth):
    return np.mean([laplacian_kernel(query, dataset[i,:], bandwidth) for i in range(dataset.shape[0])])


### Laplace kernel LSH

class LaplaceLSH:

    def __init__(self, dimension, bandwidth):
        poisson_param = dimension * 1. / bandwidth
        self.reps = np.random.poisson(poisson_param)
        self.axes = np.random.randint(0, dimension, self.reps)
        self.thresholds = np.random.uniform(0, 1, self.reps)

    def hash(self, point):
        return tuple([point[self.axes[i]] < self.thresholds[i] for i in range(self.reps)])


### Random binning features

class BinningLaplaceLSH:

    def __init__(self, dimension, bandwidth):
        self.dimension = dimension
        delta = np.random.gamma(2, bandwidth, dimension)
        self.thresholds = []
        for d in range(dimension):
            bin_length = delta[d]
            shift = np.random.uniform(0, bin_length)
            t_list = []
            while shift < 1:
                t_list.append(shift)
                shift += bin_length
            self.thresholds.append(t_list)

    def hash(self, point):
        return tuple([len([t for t in self.thresholds[d] if t < point[d]]) for d in range(self.dimension)])


### Space-efficient HBE

class FastLaplacianKDE:

    def __init__(self, dataset, bandwidth, L):
        self.n_points = dataset.shape[0]
        self.dimension = dataset.shape[1]
        self.bandwidth = bandwidth
        self.L = L

        self.sizes = np.random.binomial(self.n_points, L*1./self.n_points, self.L)
        random_samples = [np.random.choice(self.n_points, self.sizes[j], replace=False) for j in range(self.L)]

        if bandwidth >= 1:
            self.lshs = [LaplaceLSH(self.dimension, 2*self.bandwidth) for i in range(self.L)]
        else:
            self.lshs = [BinningLaplaceLSH(self.dimension, 2*self.bandwidth) for i in range(self.L)]

        self.hashed_points = []
        for j in range(L):
            bins = defaultdict(list)
            for i in random_samples[j]:
                point = dataset[i,:]
                bins[self.lshs[j].hash(point)].append(point)
            self.hashed_points.append(bins)

    def kde(self, query):
        estimators = []
        for j in range(self.L):
            query_hash = self.lshs[j].hash(query)
            bin_size = len(self.hashed_points[j][query_hash])
            if bin_size == 0:
                estimators.append(0)
            else:
                point = self.hashed_points[j][query_hash][np.random.randint(bin_size)]
                estimators.append(laplacian_kernel(query, point, 2*self.bandwidth) * bin_size * 1. / self.L)
        return np.mean(estimators)


# ### Usage example ###

# ### Generate data:
# N = 1000
# D = 100
# dataset = np.random.normal(0, 1, (N, D))
# query = dataset[np.random.randint(N), :]
# bandwidth = 2

# ### Preprocess data structure:
# L = 100
# hbe = FastLaplacianKDE(dataset, bandwidth, L)

# ### Query the data structure:
# print(query.shape)
# print("True KDE:", kde(query, dataset, bandwidth))
# print("Estimated KDE:", hbe.kde(query))