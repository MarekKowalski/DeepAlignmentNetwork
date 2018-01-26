import socket
import sys
import cv2
import struct
import numpy as np
from FaceAlignment import FaceAlignment
import utils
import time
import sys

port = 43002
nLandmarks = 51
firstLandmarkIdx = 17

def ReadNBytes(socket, nBytes):
    data = ''
    while len(data) < nBytes:
        newData = socket.recv(min(1024, nBytes - len(data)))
        if not newData:
            print "Socket disconnected"
            return

        data += newData
    return data;

def ReceiveImg(socket):
    data = ReadNBytes(sock, 8)
    if data is None:
        return None
    imageHeight, imageWidth = struct.unpack('ii', data)
    print("Receiving image, height={0}, width={1}".format(imageHeight, imageWidth))
    data = ReadNBytes(sock, imageHeight * imageWidth)
    if data is None:
        return None
    img = np.fromstring(data, dtype=np.uint8).reshape((imageHeight, imageWidth))

    return img

def ShowImageAndLandmarks(img, landmarks):
    intLandmarks = landmarks.astype(np.int32)
    for i in range(landmarks.shape[0]):
        cv2.circle(img, (intLandmarks[i][0], intLandmarks[i][1]), 2, (255, 0, 0))
        
    cv2.imshow("image", img)
    cv2.waitKey(1)

def HandleFrame(sock, model):
    data = ReadNBytes(sock, nLandmarks * 2 * 4)
    if data is None:
        return
    prevFrameLandmarks = np.fromstring(data, dtype=np.float32).reshape((-1, 2))
    prevFrameLandmarks = prevFrameLandmarks

    img = ReceiveImg(sock)
    if img is None:
        return

    landmarks, confidence = model.processNormalizedImg(img[np.newaxis])
    landmarks = landmarks[firstLandmarkIdx:]

    A, t = utils.bestFit(prevFrameLandmarks, model.initLandmarks[firstLandmarkIdx:], True)
    landmarksToSend = np.dot(landmarks, A) + t

    confidenceAndLandmarks = np.concatenate(([confidence], landmarksToSend.flatten()))
    sock.sendall(confidenceAndLandmarks.astype(np.float32).tostring())

    ShowImageAndLandmarks(img, landmarks)

def SendSettings(sock, model):
    print("Sending settings")
    sock.sendall(model.initLandmarks[17:].astype(np.float32).tostring())


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print ("Usage: python HoloFaceBackend.py IP_OF_HOLOLENS_DEVICE")
        sys.exit()

    HoloFaceIP = sys.argv[1]
    model = FaceAlignment(112, 112, 1, 1, True)
    model.loadNetwork("../DAN-Menpo-tracking.npz")

    print("Face alignment model loaded")

    while True:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2.0)
            sock.connect((HoloFaceIP, port))
            sock.settimeout(20.0)

            while True:
                try:
                    data = ReadNBytes(sock, 1)
                    if data is None:
                        continue

                    if ord(data[0]) == 0:
                        HandleFrame(sock, model)
                    elif ord(data[0]) == 1:
                        SendSettings(sock, model)
                except socket.timeout:
                    pass
        except Exception as e:
            print "No connection to HoloFace"
    
        sock.close()


