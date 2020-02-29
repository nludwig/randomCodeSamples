import sys
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

def bracketedStringToList(s):
    ss = s.split(sep = ',')
    lst = [0., 0.]
    for i, item in enumerate(ss):
        for j, c in enumerate(item):
            if c == '{':
                lst[i] = float(item[j+1:])
                break
            elif c == '}':
                lst[i] = float(item[:j])
    return lst

def loadData(fNm):
    with open(fNm, 'r') as f:
        lines = (line.strip() for line in f)
        data = [bracketedStringToList(line) for line in lines]
    return np.array(data)

def multiVariateGaussian(r, mu, sigma, sDet = None, sInv = None):
    d = r.shape[0] #dimensionality
    if sDet is None:
        sDet = np.linalg.det(sigma)
    if sInv is None:
        sInv = np.linalg.inv(sigma)
    return np.exp(-((r-mu).T @ sInv @ (r-mu))/2.) / \
           np.sqrt((2.*np.pi)**d * sDet)

def gaussianParameterEstimator(data):
    N, d = data.shape
    mu = data.mean(axis = 0)
    sigma = np.zeros((d, d))
    for i in data:
        sigma += np.einsum('i,j->ij', i-mu, i-mu)
    sigma /= N
    return mu, sigma

def empiricalDistKernelDensity(r, data, h):
    mu, det, sigma = 0., 1., np.eye(2)
    return multiVariateGaussian((r-data)/h, mu, sigma, det, sigma).mean() / h

def bootstrapSamples(data, distributionType = 'gaussian', nBootstrap = int(1e2), rng = None):
    data = np.sort(data, axis=0)
    if rng is None:
        rng = sp.random.RandomState()
    samples = np.zeros((nBootstrap,) + data.shape)
    if distributionType.lower() == 'gaussian':
        estimatedPara = gaussianParameterEstimator(data)
        samples = rng.multivariate_normal(*estimatedPara, size=(nBootstrap,data.shape[0]))
    else:
        print('{} not yet implemented'.format(distributionType), file=sys.stderr)
        exit(1)
    return samples

def andersonDarlingTest2d(data, distribution):
    n, _ = data.shape
    a2 = 0.
    for r in data:
        f = (data <= r).sum() / n
        f0 = distribution(r)
        a2 += (f - f0)**2 / (f0 * (1.-f0))
    return a2

def andersonDarlingTest2dPValue(data, distributionType = 'gaussian', nBootstrap = int(1e2), rng = None):
    if distributionType.lower() == 'gaussian':
        estimatedPara = gaussianParameterEstimator(data)
        sDet = np.linalg.det(estimatedPara[1])
        sInv = np.linalg.inv(estimatedPara[1])
        distribution = lambda r: multiVariateGaussian(r, *estimatedPara, sDet, sInv)
    else:
        print('{} not yet implemented'.format(distributionType), file=sys.stderr)
        exit(1)
    samples = bootstrapSamples(data, distributionType, nBootstrap, rng)
    a2Data = andersonDarlingTest2d(data, distribution)
    a2Samples = np.array([andersonDarlingTest2d(sample, distribution) for sample in samples])
    pValue = ((a2Samples > a2Data).sum()+1) / (nBootstrap+1)
    return pValue

def score(r):
    x, y = r
    x2 = x * x
    y2 = y * y
    if x2 + y2 < 2.5**2:
        return 30
    elif x2 + (y-5.)**2 < 2.5**2:
        return 40
    elif x2 + (y-10.)**2 < 2.5**2:
        return 50
    elif x2 + y2 < 7.5**2:
        return 20
    elif (x+10.)**2 + (y-10.)**2 < 2.5**2 or \
         (x-10.)**2 + (y-10.)**2 < 2.5**2:
        return 100
    elif x2 + y2 < 12.5**2 or \
         (x >= -12.5 and x < 12.5 and y >= 0. and y < 12.5):
        return 10
    else:
        return 0

def computeGradient(samples, sInv, r0, mu):
    scores = np.array([score(sample) for sample in samples])
    rs = np.array([sample - r0 - mu for sample in samples])
    grad = (scores[:,None] * np.einsum('ij,kj->ki', sInv, rs)).mean(axis=0)
    print(r0, grad, scores.sum()/scores.shape[0])
    return grad

def gradientDescent(r00, rng, distributionType, para, nThrows, alpha = 1e-1, eps = 1e-3):
    assert distributionType == 'gaussian' #todo: generalize!
    samples = rng.multivariate_normal(*para, size=(nThrows,))
    sInv = np.linalg.inv(para[1])
    r0 = r00
    converged = False
    while not converged:
        delta = alpha * computeGradient(samples-r0, sInv, r0, para[0])
        r0 -= delta
        converged = delta @ delta < eps * eps
    return r0

def evalOverRange(samples, sInv, r0s, mu):
    n = samples.shape[0]
    scores = np.zeros(r0s.shape[0])
    for i, r0 in enumerate(r0s):
        scores[i] = sum(score(sample+r0) for sample in samples) / n
    iMax = np.argmax(scores)
    return iMax, scores

def main():
    #parameters
    empiricalDataFNm = None
    outFNm = './emeraldOut/skeeBall.png'
    rngSeed = 123
    nBins = 200
    xBd = [-10., 10.]
    yBd = [-10., 10.]
    r00 = np.zeros(2)
    distributionType = 'gaussian'
    nThrows = int(1e3)
    alpha, eps = 1e-3, 1e-3

    #need input file with set of x, y points
    if empiricalDataFNm is not None:
        empiricalData = loadData(empiricalDataFNm)
    else:
        exit(1)

    #gaussian a good description? higher p-value better
    rng = sp.random.RandomState(seed = rngSeed)
    pValue = andersonDarlingTest2dPValue(empiricalData)
    print('multivariate gaussian p-value: {}'.format(pValue))

    #plot data + gaussian contours
    mu, sigma = gaussianParameterEstimator(empiricalData)
    sDet, sInv = np.linalg.det(sigma), np.linalg.inv(sigma)

    pSampleX = np.array([(i+0.5)*(xBd[1]-xBd[0])/nBins for i in range(0,nBins)]) + xBd[0]
    pSampleY = np.array([(i+0.5)*(yBd[1]-yBd[0])/nBins for i in range(0,nBins)]) + yBd[0]
    xv, yv = np.meshgrid(pSampleX, pSampleY, indexing='ij')
    p = np.zeros((nBins, nBins))
    for i, x in enumerate(pSampleX):
        for j, y in enumerate(pSampleY):
            r = np.array([x, y])
            p[i,j] = multiVariateGaussian(r, mu, sigma, sDet=sDet, sInv=sInv)

    plt.xlim(xBd)
    plt.ylim(yBd)
    plt.scatter(empiricalData[:,0], empiricalData[:,1])
    plt.contour(xv, yv, p)

    if outFNm is None:
        plt.show()
    else:
        plt.savefig(outFNm)

    r0l = 300
    r0xBd = [-15.0, 15.0]
    r0yBd = [-15.0, 15.0]
    samples = rng.multivariate_normal(mu, sigma, size=(nThrows,))


    r0s = np.zeros((r0l**2, 2))
    k = 0
    for i in range(r0l):
        for j in range(r0l):
            r0s[k] = np.array([(i+0.5)*(r0xBd[1]-r0xBd[0])/r0l+r0xBd[0],
                               (j+0.5)*(r0yBd[1]-r0yBd[0])/r0l+r0yBd[0]])
            k += 1
    iMax, scores = evalOverRange(samples, sInv, r0s, mu)
    print('r0Max at {} with expected score {} per throw'.format(r0s[iMax], scores[iMax]))

if __name__ == '__main__':
    main()
