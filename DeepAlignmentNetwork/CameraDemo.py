from FaceAlignment import FaceAlignment
import numpy as np
import cv2
import utils

#Change this to True if you want to use the DAN-Menpo-tracking.npz model, which is able to detect when face tracking is lost.
useTrackingModel = False

if useTrackingModel:
    model = FaceAlignment(112, 112, 1, 1, True)
    model.loadNetwork("../DAN-Menpo-tracking.npz")
else:
    model = FaceAlignment(112, 112, 1, 2)
    model.loadNetwork("../DAN-Menpo.npz")

vidIn = cv2.VideoCapture(0)
cascade = cv2.CascadeClassifier("../data/haarcascade_frontalface_alt.xml")

reset = True
landmarks = None

print ("Press space to detect the face, press escape to exit")

while True:
    vis = vidIn.read()[1]
    if len(vis.shape) > 2:
        img = np.mean(vis, axis=2).astype(np.uint8)
    else:
        img = vis.astype(np.uint8)

    if reset:
        rects = cascade.detectMultiScale(img, scaleFactor=1.2, minNeighbors=3, minSize=(50, 50)) 
        if len(rects) > 0:
            minX = rects[0][0]
            maxX = rects[0][0] + rects[0][2]
            minY = rects[0][1]
            maxY = rects[0][1] + rects[0][3]
            cv2.rectangle(vis, (minX, minY), (maxX, maxY), (255, 0, 0))
            initLandmarks = utils.bestFitRect(None, model.initLandmarks, [minX, minY, maxX, maxY])
            reset = False

            if model.confidenceLayer:
                landmarks, confidence = model.processImg(img[np.newaxis], initLandmarks)
                if confidence < 0.1:
                    reset = True
            else:
                landmarks = model.processImg(img[np.newaxis], initLandmarks)
            landmarks = landmarks.astype(np.int32)
            for i in range(landmarks.shape[0]):
                cv2.circle(vis, (landmarks[i, 0], landmarks[i, 1]), 2, (0, 255, 0))
    else:
        initLandmarks = utils.bestFitRect(landmarks, model.initLandmarks)
        if model.confidenceLayer:
            landmarks, confidence = model.processImg(img[np.newaxis], initLandmarks)
            if confidence < 0.1:
                reset = True
        else:
            landmarks = model.processImg(img[np.newaxis], initLandmarks)    
        landmarks = np.round(landmarks).astype(np.int32)
        
        for i in range(landmarks.shape[0]):
            cv2.circle(vis, (landmarks[i, 0], landmarks[i, 1]), 2, (0, 255, 0))


    cv2.imshow("image", vis)
    key = cv2.waitKey(1)

    if key == 27:
        break

    if key == 32:
        reset = True