import sys
import itertools as it
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

class WalkerInBox:
    def __init__(self, boxLength, unvisitedBias, d=2, rngSeed=123):
        self.boxLength = boxLength
        self.bias = unvisitedBias
        self.d = d
        self.nNeighbors = 3**d - 1 #d-cube with center occupied
        self.rng = sp.random.RandomState(seed=rngSeed)
        self.initializeBox()
        self.neighbors = [prod for prod in \
                          it.product((1, -1, 0), repeat=self.d) if prod != (0,0,0)]

    def initializeBox(self):
        self.visited = np.zeros(tuple(self.boxLength for _ in range(self.d)), \
                                dtype=np.bool)
        self.walker = np.array([(self.boxLength-1)//2 for _ in range(self.d)])
        self.stepsSinceVisitedNew = 0

    def inBounds(self, position):
        for pos in position:
            if pos < 0 or pos >= self.boxLength:
                return False
        return True

    def getNeighbors(self):
        neighbors = (tuple(self.walker + neighbor) for neighbor in self.neighbors)
        neighbors = [neighbor for neighbor in self.neighbors \
                     if self.inBounds(neighbor)]
        return neighbors

    def getMoveCumProbs(self, neighVisiteds):
        probs = [self.bias if neighVisited is True else 1. \
                 for neighVisited in neighVisiteds]
        s = np.sum(probs)
        probs = [prob / s for prob in probs]
        return np.cumsum(probs)

    def step(self):
        neighbors = self.getNeighbors()
        neighVisiteds = [self.visited[neighbor] for neighbor in neighbors]
        cumProbs = self.getMoveCumProbs(neighVisiteds)
        roll = self.rng.uniform()
        destinationInd = np.argmax(roll < cumProbs)
        destinationPos = neighbors[destinationInd]
        self.walker = np.array(destinationPos)
        if self.visited[destinationPos] == False:
            self.stepsSinceVisitedNew = 0
            self.visited[destinationPos] = True #visited
        else:
            self.stepsSinceVisitedNew += 1

    def nSitesVisited(self):
        return self.visited.sum()

    def fracSitesVisited(self):
        return self.nSitesVisited() / self.boxLength**self.d

def computeAvgPercentageVisited(walker, nSteps = 3*48, nEpochs = int(1e3)):
    fracVisited = [0 for _ in range(nEpochs)]
    for i in range(nEpochs):
        walker.initializeBox()
        for _ in range(nSteps):
            walker.step()
        fracVisited[i] = walker.fracSitesVisited()
    return np.mean(fracVisited)

def computeAvgTimeToVisitFrac(walker, frac = 0.5, nEpochs = int(1e3)):
    timeSteps = [0 for _ in range(nEpochs)]
    for i in range(nEpochs):
        walker.initializeBox()
        t = 0
        while walker.fracSitesVisited() < frac:
            walker.step()
            t += 1
        timeSteps[i] = t
    return np.mean(timeSteps)

def computeAvgTimeUntilTimeGapSinceLastNew(walker, timeGap = 3*24, nEpochs = int(1e3)):
    timeSteps = [0 for _ in range(nEpochs)]
    for i in range(nEpochs):
        walker.initializeBox()
        t = 0
        while walker.stepsSinceVisitedNew < timeGap:
            walker.step()
            t += 1
        timeSteps[i] = t
    return np.mean(timeSteps)

def main():
    #parameters
    boxLength = 11
    unvisitedBias = 2
    d = 2
    rngSeedUnbiased = 123
    rngSeedBiased = 1234
    filePrefix='./emeraldOut/'
    unbiasedWalkerPercentageVisitedF = 'unbiasedWalkerPercentageVisited.png'
    unbiasedWalkerTimeGapF = 'unbiasedWalkerTimeGap.png'
    biasedWalkerTimeUntilHalfVisitedF = 'biasedWalkerTimeUntilHalfVisited.png'
    biasedWalkerTimeGapF = 'biasedWalkerTimeGap.png'
    cmpWalkerTimeGapF = 'cmpWalkerTimeGap.png'
    plot = True
    largestExponent = 7

    unbiasedWalker = WalkerInBox(boxLength, 1., d=d, rngSeed=rngSeedUnbiased)
    nEpochss = [10**exponent for exponent in range(1, largestExponent)]
    percentageVisiteds = [0 for _ in range(1, largestExponent)]
    timeUntilTimeGapUnbiased = [0 for _ in range(1, largestExponent)]
    for i, nEpochs in enumerate(nEpochss):
        percentageVisiteds[i] = computeAvgPercentageVisited(unbiasedWalker, nEpochs = nEpochs)
        timeUntilTimeGapUnbiased[i] = computeAvgTimeUntilTimeGapSinceLastNew(unbiasedWalker, nEpochs=nEpochs)

    timeToVisitFrac = [0 for _ in range(1, largestExponent)]
    timeUntilTimeGapBiased = [0 for _ in range(1, largestExponent)]
    biasedWalker = WalkerInBox(boxLength, unvisitedBias, d=d, rngSeed=rngSeedBiased)
    for i, nEpochs in enumerate(nEpochss):
        timeToVisitFrac[i] = computeAvgTimeToVisitFrac(biasedWalker, nEpochs=nEpochs)
        timeUntilTimeGapBiased[i] = computeAvgTimeUntilTimeGapSinceLastNew(biasedWalker, nEpochs=nEpochs)

    if plot is True:
        plt.xlabel('Number of trials used to compute average (log scale)')
        plt.ylabel('Percentage of sites visited by unbiased walker in 3*48 timesteps')
        plt.xscale('log')
        plt.plot(nEpochss, np.array(percentageVisiteds)*100., '-', label='data')
        plt.plot(nEpochss, percentageVisiteds[-1]*np.ones_like(nEpochss)*100, label='final', linestyle='dashed')
        plt.legend()
        plt.savefig(filePrefix + unbiasedWalkerPercentageVisitedF)
        plt.clf()

        plt.xlabel('Number of trials used to compute average (log scale)')
        plt.ylabel('Time until visited half of sites for biased walker [time steps]')
        plt.xscale('log')
        plt.plot(nEpochss, timeToVisitFrac, '-', label='data')
        plt.plot(nEpochss, timeToVisitFrac[-1]*np.ones_like(nEpochss), label='final', linestyle='dashed')
        plt.legend()
        plt.savefig(filePrefix + biasedWalkerTimeUntilHalfVisitedF)
        plt.clf()

        plt.xlabel('Number of trials used to compute average (log scale)')
        plt.ylabel('Time until time gap of 3*24 timesteps [time steps]')
        plt.xscale('log')
        plt.plot(nEpochss, timeUntilTimeGapUnbiased, '-', label='unbiased')
        plt.plot(nEpochss, timeUntilTimeGapUnbiased[-1]*np.ones_like(nEpochss), label='unbiased final')
        plt.plot(nEpochss, timeUntilTimeGapBiased, '-', label='biased')
        plt.plot(nEpochss, timeUntilTimeGapBiased[-1]*np.ones_like(nEpochss), label='biased final', linestyle='dashed')
        plt.legend()
        plt.savefig(filePrefix + cmpWalkerTimeGapF)
        plt.clf()

    print('#log(# trials)\t%visitedUnbiased\ttUntilHalfVisitedBiased\ttimeUntilGapUnbiased\ttimeUntilGapBiased')
    for i in range(len(nEpochss)):
        print('{}\t{}\t{}\t{}\t{}'.format(nEpochss[i],
                                          percentageVisiteds[i]*100.,
                                          timeToVisitFrac[i],
                                          timeUntilTimeGapUnbiased[i],
                                          timeUntilTimeGapBiased[i]))

if __name__ == '__main__':
    main()
