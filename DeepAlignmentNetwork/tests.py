import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import simps
from matplotlib import pyplot as plt

def LandmarkError(imageServer, faceAlignment, normalization='centers', showResults=False, verbose=False):
    errors = []
    nImgs = len(imageServer.imgs)

    for i in range(nImgs):
        initLandmarks = imageServer.initLandmarks[i]
        gtLandmarks = imageServer.gtLandmarks[i]
        img = imageServer.imgs[i]

        if img.shape[0] > 1:
            img = np.mean(img, axis=0)[np.newaxis]

        resLandmarks = initLandmarks
        resLandmarks = faceAlignment.processImg(img, resLandmarks)

        if normalization == 'centers':
            normDist = np.linalg.norm(np.mean(gtLandmarks[36:42], axis=0) - np.mean(gtLandmarks[42:48], axis=0))
        elif normalization == 'corners':
            normDist = np.linalg.norm(gtLandmarks[36] - gtLandmarks[45])
        elif normalization == 'diagonal':
            height, width = np.max(gtLandmarks, axis=0) - np.min(gtLandmarks, axis=0)
            normDist = np.sqrt(width ** 2 + height ** 2)

        error = np.mean(np.sqrt(np.sum((gtLandmarks - resLandmarks)**2,axis=1))) / normDist       
        errors.append(error)
        if verbose:
            print("{0}: {1}".format(i, error))

        if showResults:
            plt.imshow(img[0], cmap=plt.cm.gray)            
            plt.plot(resLandmarks[:, 0], resLandmarks[:, 1], 'o')
            plt.show()

    if verbose:
        print "Image idxs sorted by error"
        print np.argsort(errors)
    avgError = np.mean(errors)
    print "Average error: {0}".format(avgError)

    return errors


def AUCError(errors, failureThreshold, step=0.0001, showCurve=False):
    nErrors = len(errors)
    xAxis = list(np.arange(0., failureThreshold + step, step))

    ced =  [float(np.count_nonzero([errors <= x])) / nErrors for x in xAxis]

    AUC = simps(ced, x=xAxis) / failureThreshold
    failureRate = 1. - ced[-1]

    print "AUC @ {0}: {1}".format(failureThreshold, AUC)
    print "Failure rate: {0}".format(failureRate)

    if showCurve:
        plt.plot(xAxis, ced)
        plt.show()

    