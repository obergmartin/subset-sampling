import itertools
import json
import math
from operator import itemgetter

import numpy as np


def getMinPairing(cur_subset: list, obs_counts: np.ndarray) -> int:
    """A way to quantify how amicable a subset is for removal."""
    subset_pairs = itertools.product(cur_subset, repeat=2)
    vals = [obs_counts[p] for p in subset_pairs]
    return min(vals)


def isPairingInSubset(pairing, subset):
    return np.intersect1d(pairing, subset).size == 2


class SubsetSampling():
    def __init__(self, pool_size, subset_size):
        self.poolSize = pool_size
        self.subsetSize = subset_size
        self.subsetMethod = None
        self.subsets = []
        self.observationCounts = self.newObservationMatrix()
        self.totPoolSubsets = math.comb(pool_size, subset_size)
        # Keep track of parameters when searching for an optimal set
        self.params = {
            "poolSize": pool_size,
            "subsetSize": subset_size,
            "method": "",
            "params": {},
            "nSubsets": 0,
            "min": np.Inf,
            "max": -np.inf
        }

    def newObservationMatrix(self):
        return np.zeros((self.poolSize, self.poolSize), dtype=int)

    def info(self):
        print(json.dumps(self.params, indent=4))

    def show(self):
        for r in self.observationCounts:
            print(','.join([f"{val:2d}" for val in r]))

    def observedPairingsMin(self):
        # use np.tril to ignore main diagonal?
        return int(self.observationCounts.min())

    def observedPairingsMax(self):
        # need int for nice info() json formatting
        return int(np.tril(self.observationCounts, k=-1).max())

    def updateObservationCounts(self, cur_subset, action="Add"):
        """Add or Remove a subset """
        if action == "Add":
            if len(cur_subset) == self.subsetSize:
                self.subsets.append(cur_subset)
                for p in itertools.product(cur_subset, repeat=2):
                    self.observationCounts[p] = self.observationCounts[p] + 1
            else:
                raise NameError('Incorrect subset size')
        elif action == "Remove":
            if cur_subset in self.subsets:
                self.subsets.pop(self.subsets.index(cur_subset))
                for p in itertools.product(cur_subset, repeat=2):
                    self.observationCounts[p] = self.observationCounts[p] - 1
            else:
                print(f"{cur_subset} not in self.subsets")
        else:
            print("Specify Add or Remove")

        self.params["nSubsets"] = len(self.subsets)
        self.params["min"] = self.observedPairingsMin()
        self.params["max"] = self.observedPairingsMax()

    def generateSteppedSubsets(self, offset=None, amount=None):
        """Create an iterator to step through all possible subset
        combinations at reqular intervals.  This works quite well
        to evenly cover the total sample space."""

        self.params["method"] = "generateSteppedSubsets"
        self.params["params"] = {
            "offset": offset,
            "amount": amount
        }

        combs = itertools.combinations(range(self.poolSize), self.subsetSize)
        if amount:
            slice_step = self.totPoolSubsets/amount
            print(f"tot {self.totPoolSubsets}, step {slice_step}")
            # subsets = list(itertools.islice(combs, offset, None, slice_step))
            ids = [int(int(i % slice_step) == 0) for i in range(self.totPoolSubsets)]
            subsets = list(itertools.compress(combs, ids))
            print(subsets)
            # subsets = [v for i, v in enumerate(combs) if i in ids]
        else:
            subsets = list(combs)

        for s in subsets:
            self.updateObservationCounts(s, "Add")

        self.params["nSubsets"] = len(self.subsets)

    def getRandomSample(self, probs=None):
        # need to remove elements that are already saturated before sampling
        samp = np.random.choice(
            self.poolSize,
            self.subsetSize,
            replace=False,
            p=probs)

        return list(samp)

    def calculateWeights(self, method="unif"):
        """"""
        if method == "unif":
            probs = [1/self.poolSize for _ in range(self.poolSize)]
        else:
            colSum = list(np.sum(self.observationCounts, axis=0))
            # add 1 to ensure there are not too many 0s and thus enough
            # elements to sample from.
            weights = max(colSum) - colSum + 0.1
            probs = [i/sum(weights) for i in weights]

        return probs

    def generateRandomSubsets(self, min_obs, n_iters=1):
        """ Finds the best list of subsets from N attempts."""

        self.params["method"] = "generateRandomSubsets"
        self.params["params"] = {
            "min_obs": min_obs
        }

        cur_best = np.inf
        for _ in range(n_iters):
            self.subsets = []
            self.observationCounts = self.newObservationMatrix()

            # Keep adding subsets untill a minimum count is achieved across
            # all pairwise observations
            while np.sum(self.observationCounts < min_obs) > 0:
                new_subset = self.getRandomSample(self.calculateWeights())
                self.updateObservationCounts(new_subset, "Add")

            self.trim(min_obs)

            # keep the best list of subsets
            if len(self.subsets) < cur_best:
                cur_best = len(self.subsets)
                best_subset_list = self.subsets

        self.subsets = []
        self.observationCounts = self.newObservationMatrix()
        for s in best_subset_list:
            self.updateObservationCounts(s, "Add")
        self.params["nSubsets"] = len(self.subsets)

    def trim(self, minVal: int) -> None:
        """Find pairings in the observation matrix that have the highest
        counts and likely belong to subsets that can be removed without
        harming the minimun count specified in [minVal]"""

        while self.observationCounts.max() > minVal:
            # Go through all subsets.  Probably not most efficient...
            # Another idea is to find the max value in the observationCounts
            # matrix and try to trim those first.
            minimum_pairing_value = [
                (si, getMinPairing(s, self.observationCounts))
                for si, s in enumerate(self.subsets)]
            minimum_pairing_value.sort(key=itemgetter(1), reverse=True)
            best_cand = minimum_pairing_value[0]
            if best_cand[1] > minVal:
                self.updateObservationCounts(self.subsets[best_cand[0]], "Remove")
            else:
                break
