import numpy as np
import json
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

def getBinaryImage(fileName):
    fJSON = open('metadata.json')
    maskJSON = json.load(fJSON)
    points = []
    for point in maskJSON[fileName]['bounds_x_y']:
        points.append((point['x'],point['y']))
    polygon = Polygon(points)
    mask = np.zeros((512,512))
    for y in range(512):
        for x in range(512):
            if polygon.contains(Point(x,y)):
                mask[y,x]=1
    return mask


def getMaxAndMin(points):
    minX = 513
    minY = 513
    maxX = -1
    maxY = -1
    for point in points:
        if point[0] > maxX:
            maxX = point[0]
        if point[0] < minX:
            minX = point[0]
        if point[1] > maxY:
            maxY = point[1]
        if point[1] < minY:
            minY = point[1]
    return((minX,minY,maxX,maxY))

def getAccuracy(mask1,mask2):
    c = 0
    for i in range(len(mask1)):
        for j in range(len(mask1[0])):
            if mask1[i,j] != mask2[i,j]:
                c += 1
    return 1 - float(float(c) / float(len(mask1)*len(mask1[0])))

def get1s(mask):
    c = 0
    for i in range(len(mask)):
        for j in range(len(mask[0])):
            if mask[i,j] == 1:
                c += 1
    return c

def get0s(mask):
    c = 0
    for i in range(len(mask)):
        for j in range(len(mask[0])):
            if mask[i,j] == 0:
                c += 1
    return c

def getRealAccuracy(realMask, predictedMask):
    True1 = 0
    True0 = 0
    False1 = 0
    False0 = 0    

    ones = get1s(realMask)
    zeroes = get0s(realMask)

    for i in range(len(realMask)):
        for j in range(len(realMask[0])):
            if realMask[i,j] == 1 and predictedMask[i,j] == 1:
                True1 += 1
            if realMask[i,j] == 0 and predictedMask[i,j] == 0:
                True0 += 1
            if realMask[i,j] == 0 and predictedMask[i,j] == 1:
                False1 += 1
            if realMask[i,j] == 1 and predictedMask[i,j] == 0:
                False0 += 1  

            T1 = True1/ones
            T0 = True0/zeroes
            F1 = False1/zeroes
            F0 = False0/ones

    return {'True1':T1,'True0':T0,'False1':F1,'False0':F0}

def visualizeError(realMask, predictedMask):
    result = np.zeros((512,512),dtype='int')
    for i in range(len(realMask)):
        for j in range(len(realMask[0])):
            if realMask[i,j] == 1 and predictedMask[i,j] == 1:
                result[i,j] = 1
            if realMask[i,j] == 0 and predictedMask[i,j] == 1:
                result[i,j] = 2
            if realMask[i,j] == 1 and predictedMask[i,j] == 0:
                result[i,j] = 3
    return result

def findCenterContour(contours):
    for i in range(len(contours)):
        cont = np.squeeze(contours[i])
        polygon = Polygon(cont)
        point = Point(256,256)
        if polygon.contains(point):
            return i
    return int(len(contours)/2)

def getBinaryFromContour(contour):
    r = np.squeeze(contour)

    MaxMin = getMaxAndMin(r)

    rectangleSize = (int(MaxMin[2])+2-int(MaxMin[0]),(int(MaxMin[3])+2-int(MaxMin[1])))
    mask = np.zeros((512,512),dtype='int')

    polygon = Polygon(r)
    for x in range(int(MaxMin[0]),int(MaxMin[2])+2):
        for y in range(int(MaxMin[1]),int(MaxMin[3])+2):
            if polygon.contains(Point(x,y)):
                xPoint = x
                yPoint = y
                mask[yPoint,xPoint]=1

    return mask

def transformToBinary(mask):
    for i in range(len(mask)):
        for j in range(len(mask[0])):
            if mask[i,j] != 0:
                mask[i,j] = 1

def getMASKfromJSON(json):
    smallMask = json['mask']
    mask = np.zeros((512,512),dtype='int')
    for y in range(len(smallMask)):
        for x in range(len(smallMask[0])):
            mask[json['minY']+y,json['minX']+x] = smallMask[y][x]
    return(mask)