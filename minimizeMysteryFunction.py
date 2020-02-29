import numpy as np
import dill as pickle

def gradient(func, x, dx=1e-3):
    #"Your code here"
    d = len(x)
    grad = [0. for _ in range(d)]
    for i in range(d):
        xHigh = list(x)
        xHigh[i] += dx / 2.
        xLow = list(x)
        xLow[i] -= dx / 2.
        grad[i] = (func(xHigh) - func(xLow)) / dx
    return grad

#step down a gradient to minimize function.
#spawns workers across the function domain
#and these workers then vote; the mode
#value is returned
def minimize(func, maxIter=1000, minChange=1e-8, stepChange=1000., dxGrad=1e-1, verbose=False, nWorkersX1=5, nWorkersX2=5, x1Min=0., x1Max=1000., x2Min=0., x2Max=1000.):
    #each worker walks down the gradient to attempt to find the minima
    def spawnMinimizeWorker(x1, x2):
        x1Old = x1
        x2Old = x2
        for i in range(maxIter):
            gradi = gradient(func, (x1Old, x2Old), dx=dxGrad)
            x1 = x1Old - stepChange*gradi[0]
            x2 = x2Old - stepChange*gradi[1]
            change = abs(x1-x1Old + x2-x2Old)
            if change < minChange: #found minima
                break
            else: #take another step
                x1Old = x1
                x2Old = x2
        return (x1, x2)

    #spawn set of workers distributed uniformly across domain
    workerDeltaX1 = (x1Max-x1Min) / nWorkersX1
    x1StartingCoords = [i*workerDeltaX1 for i in range(nWorkersX1)]
    workerDeltaX2 = (x2Max-x2Min) / nWorkersX2
    x2StartingCoords = [i*workerDeltaX2 for i in range(nWorkersX2)]

    #gather worker results
    workerOut = []
    for x1_0 in x1StartingCoords:
        for x2_0 in x2StartingCoords:
            workerOut.append(spawnMinimizeWorker(x1_0, x2_0))

    #vote for "numerical" mode
    nSolutions = len(workerOut)
    votes = [0 for _ in range(nSolutions)]
    i = 0
    while i < nSolutions:
        j = i+1
        while j < nSolutions:
            if isClose(workerOut[i][0], workerOut[j][0]) and isClose(workerOut[i][1], workerOut[j][1]):
                votes[i] += 1
                del(workerOut[j])
                del(votes[j])
                nSolutions -= 1
            else:
                j += 1
        i += 1
    if verbose is True:
        print(workerOut)
        print(votes)

    voteWinner = argmax(votes)
    return workerOut[voteWinner]

def isClose(x, y, delta=1e-1):
    if abs(x-y) < delta:
        return True
    else:
        return False

def argmax(x):
    xmax = max(x)
    for i in range(len(x)):
        if x[i] == xmax:
            return i

if __name__ == '__main__':
    with open('f.pkl', 'rb') as file:
        f = pickle.load(file)
    print(minimize(f, verbose=True))
