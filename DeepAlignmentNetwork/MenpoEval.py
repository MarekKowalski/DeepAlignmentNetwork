from FaceAlignment import FaceAlignment
import utils
import numpy as np
import os
import glob
import cv2
import ntpath
from matplotlib import pyplot as plt

ptsOutputDir = "../results/pts/"
imgOutputDir = "../results/imgs/"
MenpoDir = "../data/images/Menpo testset/semifrontal/"
imageHeightFraction = 0.46

networkFilename = "../DAN-Menpo.npz"
network = FaceAlignment(112, 112, 1, nStages=2)
network.loadNetwork(networkFilename)

print "Image height fraction: " + str(imageHeightFraction)

if not os.path.exists(ptsOutputDir):
    os.makedirs(ptsOutputDir)
if not os.path.exists(imgOutputDir):
    os.makedirs(imgOutputDir)
filenames = glob.glob(MenpoDir + "\\*.*")


for i in range(len(filenames)):
    print(i)

    img = cv2.imread(filenames[i])
    imgColor = np.copy(img[:, :, [2, 1, 0]])
    if len(img.shape) > 2:
        img = np.mean(img, axis=2)

    faceHeight = img.shape[0] * imageHeightFraction
    faceWidth = faceHeight
    center = np.array(img.shape) / 2
        
    box = [center[1] - faceWidth / 2, center[0] - faceHeight / 2, center[1] + faceWidth / 2, center[0] + faceHeight / 2]

    #first step
    initLandmarks = utils.bestFitRect([], network.initLandmarks, box)
    firstStepLandmarks = network.processImg(img[np.newaxis], initLandmarks)

    #second step
    normImg, transform = network.CropResizeRotate(img[np.newaxis], firstStepLandmarks)
    normFirstStepLandmarks = np.dot(firstStepLandmarks, transform[0]) + transform[1]
    initLandmarks2 = utils.bestFitRect(normFirstStepLandmarks, network.initLandmarks)

    finalLandmarks = network.processImg(normImg, initLandmarks2)
    finalLandmarks = np.dot(finalLandmarks - transform[1], np.linalg.inv(transform[0]))


    baseName = ntpath.basename(filenames[i])[:-4]    
    utils.saveToPts(ptsOutputDir + baseName + ".pts", finalLandmarks)
        
    plt.plot((box[2], box[0], box[0], box[2], box[2]), (box[1], box[1], box[3], box[3], box[1]), 'b', linewidth=3.0)
    plt.plot(finalLandmarks[:, 0], finalLandmarks[:, 1], 'go')
    plt.imshow(imgColor, cmap=plt.cm.gray)
    plt.savefig(imgOutputDir + baseName + ".png", dpi=200)
    plt.clf()
