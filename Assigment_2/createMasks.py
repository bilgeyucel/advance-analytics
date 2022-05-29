from torch.utils.data import Dataset
from os import listdir
from os.path import isfile, join
from matplotlib import pyplot as plt
import json
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import cv2
import numpy as np
from functions import *

#Image Folder
IMAGE_PATH = 'images'
MASK_PATH = 'metadata.json'
imageFiles = [f for f in listdir(IMAGE_PATH) if isfile(join(IMAGE_PATH, f))]

#Metadata FOlder
f = open(MASK_PATH)
maskJSON = json.load(f)

idx = 0

#Points
points = []
for point in maskJSON[imageFiles[idx]]['bounds_x_y']:
    points.append((point['x'],point['y']))
MaxMin = getMaxAndMin(points)

rectangleSize = (int(MaxMin[2])+2-int(MaxMin[0]),(int(MaxMin[3])+2-int(MaxMin[1])))
mask = np.zeros(rectangleSize)

polygon = Polygon(points)
for x in range(int(MaxMin[0]),int(MaxMin[2])+2):
    for y in range(int(MaxMin[1]),int(MaxMin[3])+2):
        if polygon.contains(Point(x,y)):
            mask[x-int(MaxMin[0]),y-int(MaxMin[1])]=1

image = cv2.imread(join(IMAGE_PATH,imageFiles[idx]))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(15,15))
fig.add_subplot(1,2,1)
plt.imshow(image)
fig.add_subplot(1,2,2)
plt.imshow(mask)

# i = 400
# createdFiles = []
# fileOut = fileNames[i][:-4]+'.out'
# if fileOut not in createdFiles:
#     points = []
#     for point in maskJSON[fileNames[i]]['bounds_x_y']:
#         points.append((point['x'],point['y']))
#     polygon = Polygon(points)
#     mask = np.zeros((512,512))
#     for y in range(512):
#         for x in range(512):
#             if polygon.contains(Point(x,y)):
#                 mask[y,x]=1
#     saveMask(mask,fileNames[i])