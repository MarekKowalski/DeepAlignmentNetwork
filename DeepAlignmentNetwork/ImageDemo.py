from FaceAlignment import FaceAlignment
import numpy as np
import cv2
import utils

model = FaceAlignment(112, 112, 1, 1, True)
model.loadNetwork("../data/DAN-Menpo-tracking.npz")

cascade = cv2.CascadeClassifier("../data/haarcascade_frontalface_alt.xml")

color_img = cv2.imread("../data/jk.jpg")
if len(color_img.shape) > 2:
    gray_img = np.mean(color_img, axis=2).astype(np.uint8)
else:
    gray_img = color_img.astype(np.uint8)

# reset = True
landmarks = None

# if reset:
rects = cascade.detectMultiScale(gray_img, scaleFactor=1.2, minNeighbors=3, minSize=(50, 50))

for rect in rects:
    tl_x = rect[0]
    tl_y = rect[1]
    br_x = tl_x + rect[2]
    br_y = tl_y + rect[3]

    cv2.rectangle(color_img, (tl_x, tl_y), (br_x, br_y), (255, 0, 0))

    initLandmarks = utils.bestFitRect(None, model.initLandmarks, [tl_x, tl_y, br_x, br_y])

    if model.confidenceLayer:
        landmarks, confidence = model.processImg(gray_img[np.newaxis], initLandmarks)
        if confidence < 0.1:
            reset = True
    else:
        landmarks = model.processImg(gray_img[np.newaxis], initLandmarks)

    landmarks = landmarks.astype(np.int32)
    for i in range(landmarks.shape[0]):
        cv2.circle(color_img, (landmarks[i, 0], landmarks[i, 1]), 2, (0, 255, 0))

cv2.imshow("image", color_img)

key = cv2.waitKey(0)
