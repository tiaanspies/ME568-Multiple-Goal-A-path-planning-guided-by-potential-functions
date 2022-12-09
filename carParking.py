import numpy as np
import math
from matplotlib import pyplot as plt

import me570_geometry as geo

def generateHalfParking(startX, startY, colNr, length, width, linewidth):
    """Makes a parking polygon that faces only one direction"""

    # generate repeating part of the pattern
    # generate goals in centre of parking spots
    vertices = np.zeros(shape=(2,0), dtype=np.float64)
    goals = np.zeros(shape=(2,0), dtype=np.float64)
    for i in range(colNr):
        xPos = i*(width + linewidth) + startX
        newVerts = np.array([
                [xPos, xPos, xPos+linewidth, xPos+linewidth, xPos+linewidth+width],
                [startY,startY+length, startY+length, startY, startY]
            ])
        vertices = np.hstack([vertices, newVerts])

        goal = np.array([[xPos+linewidth+width/2], [startY+length/2]])
        goals = np.hstack([goals, goal])

    # add last section that is not repeating
    newVerts = np.array([
            [xPos+linewidth+width, xPos+2*linewidth+width,  xPos+2*linewidth+width],
            [startY+length, startY+length, startY-linewidth]
        ])
    vertices = np.hstack([vertices, newVerts])
    
    # move the first vertex down so that last line joins correctly
    vertices[1, 0] = startY-linewidth
    flipped = np.flip(vertices.copy(), axis = 1)
    # flipped = vertices.copy()
    return flipped, goals

def generateMirroredParking(startX, startY, colNr, length, width, lineWidth):
    """Makes a double direction parking polygon"""
    # generate repeating pattern
    vertices = np.zeros(shape=(2,0), dtype=np.float64)
    goals = np.zeros(shape=(2,0), dtype=np.float64)

    # generate upper half of parking spots with goals in centre
    for i in range(colNr):
        xPos = i*(width + lineWidth)
        newVerts = np.array([
                [xPos, xPos, xPos+lineWidth, xPos+lineWidth, xPos+lineWidth+width],
                [lineWidth/2, length, length, lineWidth/2, lineWidth/2]
            ])
        vertices = np.hstack([vertices, newVerts])

        goal = np.array([[xPos+lineWidth+width/2], [length/2+lineWidth/2]])
        goals = np.hstack([goals, goal])


    # generate last edge
    newVerts = np.array([
            [xPos+lineWidth+width, xPos+2*lineWidth+width],
            [length, length]
        ])
    vertices = np.hstack([vertices, newVerts])

    # create duplicates and flip them around the x axis
    # also flips the order so that the polygon is defined 
    # counter clockwise
    vertFlip = np.flip(vertices.copy(), axis=1)
    vertFlip[1, :] *= -1

    goalsFlip = np.flip(goals.copy(), axis=1)
    goalsFlip[1, :] *= -1
    
    # join flipped pairs
    vertices = np.hstack([vertices, vertFlip])
    goals = np.hstack([goals, goalsFlip])

    # translate to requested postions
    vertices[0, :] += startX
    vertices[1, :] += startY

    goals[0, :] += startX
    goals[1, :] += startY
    
    return np.flip(vertices, axis=1), goals

class Car():
    def __init__(self, length, width, steeringAngle) -> None:
        self.poly =geo.Polygon(np.array([[0, length, length, 0],
            [width/2, width/2, -width/2, -width/2]]))

        self.steerAngle = steeringAngle
        self.width = width

        self.validThetas = np.linspace(0, np.pi*2, 6)

    def transform(self, thetas, Xs):
        # transform the car into new position
        polygons = np.zeros(shape=(0, 2, 4), dtype=np.float64)
        for theta, X in zip(thetas, Xs.T):
            R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            verts = R @ self.poly.vertices + np.reshape(X, (2,1))

            polygons = np.append(polygons, [verts], axis=0)

        return polygons

    def plot(self, thetas, Xs):
        newVerts = self.transform(thetas, Xs)

        for poly in newVerts:
            x = np.hstack([poly[0, :], poly[0, 0]])
            y = np.hstack([poly[1, :], poly[1, 0]])
            
            plt.plot(x, y, 'g-')
            plt.plot(poly[0, 1:3], poly[1, 1:3], 'r-')

        plt.plot(Xs[0, :], Xs[1, :], ".b")

    def possibleNextNodes(self, X, travelDist):

        beta = travelDist/self.width * self.steerAngle
        R = travelDist/beta

        # Order of nodes is:
        # 0: Forward left
        # 1: Forward
        # 2: Forward Right
        # 3: Back left
        # 4: Backwards
        # 5: Back right

        x = R * (math.sin(X[2] + beta) + math.sin(X[2]))
        y = R * (math.cos(X[2]) - math.cos(X[2] + beta))

        xZero  = math.cos(X[2])*travelDist
        yZero = math.sin(X[2])*travelDist

        XDiff = np.array([[x, xZero, x, -x, -xZero, -x],
                        [y, yZero, -y, y, yZero, -y],
                        [beta, 0, -beta, -beta, 0, beta]])

        nextPos = X + XDiff

        # thetas = np.array(
        #         [theta+beta, theta, theta-beta, theta-beta, theta, theta+beta]
        #     ) % (2*np.pi)

        return nextPos

            