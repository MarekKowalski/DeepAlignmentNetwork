import numpy as np

def loadFromPts(filename):
    landmarks = np.genfromtxt(filename, skip_header=3, skip_footer=1)
    landmarks = landmarks - 1

    return landmarks

def saveToPts(filename, landmarks):
    pts = landmarks + 1

    header = 'version: 1\nn_points: {}\n{{'.format(pts.shape[0])
    np.savetxt(filename, pts, delimiter=' ', header=header, footer='}', fmt='%.3f', comments='')

def bestFitRect(points, meanS, box=None):
    if box is None:
        box = np.array([points[:, 0].min(), points[:, 1].min(), points[:, 0].max(), points[:, 1].max()])
    boxCenter = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2 ])

    boxWidth = box[2] - box[0]
    boxHeight = box[3] - box[1]

    meanShapeWidth = meanS[:, 0].max() - meanS[:, 0].min()
    meanShapeHeight = meanS[:, 1].max() - meanS[:, 1].min()

    scaleWidth = boxWidth / meanShapeWidth
    scaleHeight = boxHeight / meanShapeHeight
    scale = (scaleWidth + scaleHeight) / 2

    S0 = meanS * scale

    S0Center = [(S0[:, 0].min() + S0[:, 0].max()) / 2, (S0[:, 1].min() + S0[:, 1].max()) / 2]    
    S0 += boxCenter - S0Center

    return S0

def bestFit(destination, source, returnTransform=False):
    destMean = np.mean(destination, axis=0)
    srcMean = np.mean(source, axis=0)

    srcVec = (source - srcMean).flatten()
    destVec = (destination - destMean).flatten()

    a = np.dot(srcVec, destVec) / np.linalg.norm(srcVec)**2
    b = 0
    for i in range(destination.shape[0]):
        b += srcVec[2*i] * destVec[2*i+1] - srcVec[2*i+1] * destVec[2*i] 
    b = b / np.linalg.norm(srcVec)**2
    
    T = np.array([[a, b], [-b, a]])
    srcMean = np.dot(srcMean, T)

    if returnTransform:
        return T, destMean - srcMean
    else:
        return np.dot(srcVec.reshape((-1, 2)), T) + destMean

def mirrorShape(shape, imgShape=None):
    imgShapeTemp = np.array(imgShape)
    shape2 = mirrorShapes(shape.reshape((1, -1, 2)), imgShapeTemp.reshape((1, -1)))[0]

    return shape2

def mirrorShapes(shapes, imgShapes=None):
    shapes2 = shapes.copy()
    
    for i in range(shapes.shape[0]):
        if imgShapes is None:
            shapes2[i, :, 0] = -shapes2[i, :, 0]
        else:
            shapes2[i, :, 0] = -shapes2[i, :, 0] + imgShapes[i][1]
        
        lEyeIndU = range(36, 40)
        lEyeIndD = [40, 41]
        rEyeIndU = range(42, 46)
        rEyeIndD = [46, 47]
        lBrowInd = range(17, 22)
        rBrowInd = range(22, 27)
        
        uMouthInd = range(48, 55)
        dMouthInd = range(55, 60)
        uInnMouthInd = range(60, 65)
        dInnMouthInd = range(65, 68)
        noseInd = range(31, 36)
        beardInd = range(17)
         
        lEyeU = shapes2[i, lEyeIndU].copy()
        lEyeD = shapes2[i, lEyeIndD].copy()
        rEyeU = shapes2[i, rEyeIndU].copy()       
        rEyeD = shapes2[i, rEyeIndD].copy() 
        lBrow = shapes2[i, lBrowInd].copy()
        rBrow = shapes2[i, rBrowInd].copy()

        uMouth = shapes2[i, uMouthInd].copy()
        dMouth = shapes2[i, dMouthInd].copy()
        uInnMouth = shapes2[i, uInnMouthInd].copy()
        dInnMouth = shapes2[i, dInnMouthInd].copy()
        nose = shapes2[i, noseInd].copy()
        beard = shapes2[i, beardInd].copy()
        
        lEyeIndU.reverse()
        lEyeIndD.reverse()
        rEyeIndU.reverse()
        rEyeIndD.reverse()
        lBrowInd.reverse()
        rBrowInd.reverse()
        
        uMouthInd.reverse()
        dMouthInd.reverse()
        uInnMouthInd.reverse()
        dInnMouthInd.reverse()
        beardInd.reverse()
        noseInd.reverse()   
        
        shapes2[i, rEyeIndU] = lEyeU
        shapes2[i, rEyeIndD] = lEyeD
        shapes2[i, lEyeIndU] = rEyeU
        shapes2[i, lEyeIndD] = rEyeD
        shapes2[i, rBrowInd] = lBrow
        shapes2[i, lBrowInd] = rBrow
        
        shapes2[i, uMouthInd] = uMouth
        shapes2[i, dMouthInd] = dMouth
        shapes2[i, uInnMouthInd] = uInnMouth
        shapes2[i, dInnMouthInd] = dInnMouth
        shapes2[i, noseInd] = nose
        shapes2[i, beardInd] = beard
        
    return shapes2

