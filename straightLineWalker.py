import sys
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

class StraightLineWalker:
    def __init__(self, initPos = None, initOri = None, rngSeed = 123, v = 0.25, boxSize = np.array([10., 12.])/3.28084, \
                 boxConstraints = np.array([[[0., 8.], [3.28084/np.sqrt(2), 3.28084/np.sqrt(2)], [4., 12.]]])/3.28084, \
                 walkerRadius = 0.5/3.28084, gridBinsPerDistance = 100.):
        self.rng = sp.random.RandomState(seed = rngSeed)
        self.v = v
        self.r = walkerRadius
        self.boxSize = boxSize
        self.gtMaxLen = 2. * self.boxSize @ self.boxSize
        self.buildBoxConstraints(boxConstraints)
        self.gridBinsPerDistance = gridBinsPerDistance
        self.reset(initPos, initOri)

    def reset(self, initPos = None, initOri = None):
        if initPos is None:
            self.pos = self.boxSize / 2.
        else:
            self.pos = initPos
        if initOri is None:
            self.randomOri()
        else:
            self.ori = initOri
        self.posTrajectory = [self.pos]
        self.simulationTime = 0.
        self.resetGrid()

    def resetGrid(self):
        gridShape = (self.boxSize * self.gridBinsPerDistance + 1.).astype(np.int32)
        self.gridVisited = np.zeros(gridShape, dtype = np.int16)
        self.gridPos = np.zeros((*gridShape, 2))
        for i in range(gridShape[0]):
            for j in range(gridShape[1]):
                self.gridPos[i,j] = ((i+0.5)*self.boxSize[0]/gridShape[0], \
                                     (j+0.5)*self.boxSize[1]/gridShape[1])
        r0, v, r1 = self.boxConstraints[0] #assumes default case; todo: generalize!
        u = self.makeOrthogonal(v)
        self.updateGrid(r0, r1, u, v)

    def updateGrid(self, r0, r1, u, v):
        r0uv = self.changeBases(r0, u, v)
        r1uv = self.changeBases(r1, u, v)
        for i in range(self.gridVisited.shape[0]):
            for j in range(self.gridVisited.shape[1]):
                Ruv = self.changeBases(self.gridPos[i, j], u, v)
                if Ruv[0] >= r0uv[0]:
                     self.gridVisited[i, j] = -1

    def buildBoxConstraints(self, boxConstraints):
        l = boxConstraints.shape[0]
        self.nWalls = l + 4
        self.boxConstraints = np.zeros((self.nWalls, 3, 2))
        self.boxConstraints[:l] = boxConstraints[:l]
        self.boxConstraints[l] =   [[self.boxSize[0], 0.],
                                    [0., 1.],
                                    [*self.boxSize]]
        self.boxConstraints[l+1] = [[*self.boxSize],
                                    [-1., 0.],
                                    [0., self.boxSize[1]]]
        self.boxConstraints[l+2] = [[0., self.boxSize[1]],
                                    [0., -1.],
                                    [0., 0.]]
        self.boxConstraints[l+3] = [[0., 0.],
                                    [1., 0.],
                                    [self.boxSize[0], 0.]]

    def randomOri(self):
        angle = self.rng.uniform() * 2. * np.pi
        self.ori = np.array([np.cos(angle), np.sin(angle)])

    def computeIntersection(self, linePos, lineOri):
        if abs(self.ori @ lineOri) == 1.:
            return None
        rx, ry, vx, vy = *self.pos, *self.ori
        sx, sy, wx, wy = *linePos, *lineOri
        if wy == 0.:
            x = (sy - ry + wy/wx*(rx - sx)) / (vy - wy/wx*vx)
        else:
            try:
                x = (sx - rx + wx/wy*(ry - sy)) / (vx - wx/wy*vy)
            except ZeroDivisionError:
                return None
        return self.pos + self.ori * x

    def pointInBounds(self, point, lower, ori, upper):
        return (point - lower) @ ori >= 0. \
            and (upper - point) @ ori >= 0.

    def distFrom(self, point):
        dif = self.pos - point
        return np.sqrt(dif @ dif)

    def findWallCollision(self):
        collisionPoint = [[] for i in range(self.nWalls)]
        for i in range(self.nWalls):
            point = self.computeIntersection(*self.boxConstraints[i, :-1])
            if point is not None:
                if self.pointInBounds(point, *self.boxConstraints[i]) and \
                        (point - self.pos) @ self.ori > 0. :
                    collisionPoint[i] = point
        if sum(len(item) for item in collisionPoint) == 0:
            return None
        dists = [self.distFrom(point) if len(point) > 0 else self.gtMaxLen \
                 for point in collisionPoint]
        collisionWall = np.argmin(dists)
        #print(self.pos, self.ori, collisionPoint)
        return np.array(collisionPoint[collisionWall])

    def walkerPosFromWallPoint(self, wallPoint):
        d = self.r
        cos, sin = self.ori
        theta = np.arctan2(self.ori[1], self.ori[0]) / np.pi
        if (theta >= 0. and theta < 1./4.) \
         or (theta >= 7./4. and theta < 2.):
            d /= cos
        elif theta >= 1./4. and theta < 3./4.:
            d /= sin
        elif theta >= 3./4. and theta < 5./4.:
            d /= -cos
        elif theta >= 5./4. and theta < 7./4.:
            d /= -sin
        return wallPoint - d*self.ori

    def buildPath(self):
        wallPoint = self.findWallCollision()
        if wallPoint is not None:
            newPos = self.walkerPosFromWallPoint(wallPoint)
            #print(wallPoint, newPos)
            dist = self.pos - newPos
            dist = np.sqrt(dist @ dist)
            self.simulationTime += dist / self.v
            self.pos = newPos
            self.posTrajectory.append(self.pos)
        self.updateGridAfterMove()
        self.randomOri()

    def makeOrthogonal(self, v): #rotate ard z-axis
        return np.array([v[1], -v[0]])

    def changeBases(self, R, u, v):
        Rx, Ry = R
        if v[1] != 0.:
            Rv = (Ry + Rx*v[0]/v[1]) / (v[1] - v[0]*v[0]/v[1])
            Ru = (Rx - Rv*v[0]) / v[1]
        else:
            Rv = (Rx + Ry*v[1]/v[0]) / (v[0] + v[1]*v[1]/v[0])
            Ru = (Rv*v[1] - Ry) / v[0]
        return Ru, Rv

    def updateGridAfterMove(self):
        r0, r1 = self.posTrajectory[-2:]
        v = self.ori
        u = self.makeOrthogonal(v)
        r0uv = self.changeBases(r0, u, v)
        r1uv = self.changeBases(r1, u, v)
        for i in range(self.gridVisited.shape[0]):
            for j in range(self.gridVisited.shape[1]):
                Ruv = self.changeBases(self.gridPos[i, j], u, v)
                if Ruv[0] >= r0uv[0] - self.r and Ruv[0] <= r0uv[0] + self.r \
                    and Ruv[1] >= r0uv[1] and Ruv[1] <= r1uv[1]:
                     self.gridVisited[i, j] = 1

    def plotTrajectory(self, fOut = None):
        #build box
        x = [i for i in range(int(self.boxSize[0]+1.))]
        plt.plot(x, np.zeros_like(x))
        plt.xlim([-0.5, self.boxSize[0]+0.5])
        plt.ylim([-0.5, self.boxSize[1]+0.5])

        bdColor = 'blue'
        plotLine([self.boxSize[0], 0.], self.boxSize, color = bdColor)
        plotLine(self.boxSize, [0., self.boxSize[1]], color = bdColor)
        plotLine([0., self.boxSize[1]], [0., 0.], color = bdColor)
        plotLine([0., 0.], [self.boxSize[0], 0.], color = bdColor)
        plotLine(self.boxConstraints[0, 0], self.boxConstraints[0, 2], color = bdColor) #specific to default case; todo: generalize!

        #draw trajectory
        fig = plt.gcf()
        ax = fig.gca()
        walkerStart = plt.Circle(tuple(self.posTrajectory[0]), self.r)
        ax.add_artist(walkerStart)
        for i in range(len(self.posTrajectory)-1):
            plt.arrow(*self.posTrajectory[i], *(self.posTrajectory[i+1]-self.posTrajectory[i]), length_includes_head = True, head_width = 0.05, head_length = 0.15)
        if fOut is None:
            plt.show()
        else:
            plt.savefig(fOut)

def plotLine(p1, p2, color = None):
    x = [p1[0], p2[0]]
    y = [p1[1], p2[1]]
    plt.plot(x, y, color = color)

def walkForTime(walker, timeToWalk):
    walker.reset()
    while walker.simulationTime < timeToWalk:
        walker.buildPath()
    visitedFrac = (walker.gridVisited > 0).sum() / (walker.gridVisited >= 0).sum()
    return visitedFrac

def computeAvgTimeToVisitFrac(walker, frac = 0.95, nEpochs = int(1e3)):
    timeSteps = [0 for _ in range(nEpochs)]
    for i in range(nEpochs):
        walker.reset()
        while (walker.gridVisited > 0).sum() / (walker.gridVisited >= 0).sum() < frac:
            walker.buildPath()
        timeSteps[i] = walker.simulationTime
    return np.mean(timeSteps)

def main():
    #parameters
    timeToWalk = 10. * 60.
    plot = True
    pathOut = './emeraldOut/roombaPath.png'
    largestExponent = 3
    timeToVisitFracPlotOut = './emeraldOut/roombaTimeToVisitFrac-3.png'

    walker = StraightLineWalker()

    visitedFrac = walkForTime(walker, timeToWalk)
    print('#covered approximately {}% of room area'.format(visitedFrac * 100.))
    if plot is True:
        walker.plotTrajectory(fOut = pathOut)

    nEpochss = [10**i for i in range(1, largestExponent)]
    timeToVisitFrac = [computeAvgTimeToVisitFrac(walker, nEpochs=nEpochs) 
                       for nEpochs in nEpochss]
    if plot is True:
        plt.clf()
        plt.xlabel('Number of trials used to compute average')
        plt.ylabel('Average ime to visit 95% of room area')
        plt.xscale('log')
        plt.plot(nEpochss, np.array(timeToVisitFrac), label='data')
        plt.plot(nEpochss, timeToVisitFrac[-1]*np.ones_like(nEpochss), label='final', linestyle='dashed')
        plt.legend()
        plt.savefig(timeToVisitFracPlotOut)

    print('#log(# trials)\ttime to visit')
    for i in range(len(nEpochss)):
        print('{}\t{}'.format(nEpochss[i], timeToVisitFrac[i]))

if __name__ == '__main__':
    main()
