from __future__ import print_function

import lasagne
from lasagne.layers import Conv2DLayer, batch_norm
from lasagne.init import GlorotUniform

import numpy as np
import theano

from scipy import ndimage

from AffineTransformLayer import AffineTransformLayer
from TransformParamsLayer import TransformParamsLayer
from LandmarkImageLayer import LandmarkImageLayer
from LandmarkInitLayer import LandmarkInitLayer
from LandmarkTranformLayer import LandmarkTransformLayer

import utils

class FaceAlignment(object):
    def __init__(self, height, width, nChannels, nStages, confidenceLayer=False):        
        self.landmarkPatchSize = 16

        self.data = theano.tensor.tensor4('inputs', dtype=theano.config.floatX)
        self.targets = theano.tensor.tensor4('targets')

        self.imageHeight = height
        self.imageWidth = width
        self.nChannels = nChannels

        self.errors = []
        self.errorsTrain = []

        self.nStages = nStages
        self.confidenceLayer = confidenceLayer

    def initializeNetwork(self):
        self.layers = self.createCNN()
        self.network = self.layers['output']
        
        self.prediction = lasagne.layers.get_output(self.network, deterministic=True)
        self.generate_network_output = theano.function([self.data], [self.prediction])    

    def addDANStage(self, stageIdx, net):
        prevStage = 's' + str(stageIdx - 1)
        curStage = 's' + str(stageIdx)

        #CONNNECTION LAYERS OF PREVIOUS STAGE
        net[prevStage + '_transform_params'] = TransformParamsLayer(net[prevStage + '_landmarks'], self.initLandmarks)
        net[prevStage + '_img_output'] = AffineTransformLayer(net['input'], net[prevStage + '_transform_params'])    
            
        net[prevStage + '_landmarks_affine'] = LandmarkTransformLayer(net[prevStage + '_landmarks'], net[prevStage + '_transform_params'])
        net[prevStage + '_img_landmarks'] = LandmarkImageLayer(net[prevStage + '_landmarks_affine'], (self.imageHeight, self.imageWidth), self.landmarkPatchSize)

        net[prevStage + '_img_feature'] = lasagne.layers.DenseLayer(net[prevStage + '_fc1'], num_units=56 * 56, W=GlorotUniform('relu'))
        net[prevStage + '_img_feature'] = lasagne.layers.ReshapeLayer(net[prevStage + '_img_feature'], (-1, 1, 56, 56))
        net[prevStage + '_img_feature'] = lasagne.layers.Upscale2DLayer(net[prevStage + '_img_feature'], 2)

        #CURRENT STAGE
        net[curStage + '_input'] = batch_norm(lasagne.layers.ConcatLayer([net[prevStage + '_img_output'], net[prevStage + '_img_landmarks'], net[prevStage + '_img_feature']], 1))

        net[curStage + '_conv1_1'] = batch_norm(Conv2DLayer(net[curStage + '_input'], 64, 3, pad='same', W=GlorotUniform('relu')))
        net[curStage + '_conv1_2'] = batch_norm(Conv2DLayer(net[curStage + '_conv1_1'], 64, 3, pad='same', W=GlorotUniform('relu')))
        net[curStage + '_pool1'] = lasagne.layers.Pool2DLayer(net[curStage + '_conv1_2'], 2)

        net[curStage + '_conv2_1'] = batch_norm(Conv2DLayer(net[curStage + '_pool1'], 128, 3, pad=1, W=GlorotUniform('relu')))
        net[curStage + '_conv2_2'] = batch_norm(Conv2DLayer(net[curStage + '_conv2_1'], 128, 3, pad=1, W=GlorotUniform('relu')))
        net[curStage + '_pool2'] = lasagne.layers.Pool2DLayer(net[curStage + '_conv2_2'], 2)

        net[curStage + '_conv3_1'] = batch_norm (Conv2DLayer(net[curStage + '_pool2'], 256, 3, pad=1, W=GlorotUniform('relu')))
        net[curStage + '_conv3_2'] = batch_norm (Conv2DLayer(net[curStage + '_conv3_1'], 256, 3, pad=1, W=GlorotUniform('relu')))  
        net[curStage + '_pool3'] = lasagne.layers.Pool2DLayer(net[curStage + '_conv3_2'], 2)
        
        net[curStage + '_conv4_1'] = batch_norm(Conv2DLayer(net[curStage + '_pool3'], 512, 3, pad=1, W=GlorotUniform('relu')))
        net[curStage + '_conv4_2'] = batch_norm (Conv2DLayer(net[curStage + '_conv4_1'], 512, 3, pad=1, W=GlorotUniform('relu')))  
        net[curStage + '_pool4'] = lasagne.layers.Pool2DLayer(net[curStage + '_conv4_2'], 2)
        
        net[curStage + '_pool4'] = lasagne.layers.FlattenLayer(net[curStage + '_pool4'])           
        net[curStage + '_fc1_dropout'] = lasagne.layers.DropoutLayer(net[curStage + '_pool4'], p=0.5)
       
        net[curStage + '_fc1'] = batch_norm(lasagne.layers.DenseLayer(net[curStage + '_fc1_dropout'], num_units=256, W=GlorotUniform('relu')))

        net[curStage + '_output'] = lasagne.layers.DenseLayer(net[curStage + '_fc1'], num_units=136, nonlinearity=None)
        net[curStage + '_landmarks'] = lasagne.layers.ElemwiseSumLayer([net[prevStage + '_landmarks_affine'], net[curStage + '_output']])

        net[curStage + '_landmarks'] = LandmarkTransformLayer(net[curStage + '_landmarks'], net[prevStage + '_transform_params'], True)

    def createCNN(self):
        net = {}
        net['input'] = lasagne.layers.InputLayer(shape=(None, self.nChannels, self.imageHeight, self.imageWidth), input_var=self.data)       
        print("Input shape: {0}".format(net['input'].output_shape))

        #STAGE 1
        net['s1_conv1_1'] = batch_norm(Conv2DLayer(net['input'], 64, 3, pad='same', W=GlorotUniform('relu')))
        net['s1_conv1_2'] = batch_norm(Conv2DLayer(net['s1_conv1_1'], 64, 3, pad='same', W=GlorotUniform('relu')))
        net['s1_pool1'] = lasagne.layers.Pool2DLayer(net['s1_conv1_2'], 2)

        net['s1_conv2_1'] = batch_norm(Conv2DLayer(net['s1_pool1'], 128, 3, pad=1, W=GlorotUniform('relu')))
        net['s1_conv2_2'] = batch_norm(Conv2DLayer(net['s1_conv2_1'], 128, 3, pad=1, W=GlorotUniform('relu')))
        net['s1_pool2'] = lasagne.layers.Pool2DLayer(net['s1_conv2_2'], 2)

        net['s1_conv3_1'] = batch_norm (Conv2DLayer(net['s1_pool2'], 256, 3, pad=1, W=GlorotUniform('relu')))
        net['s1_conv3_2'] = batch_norm (Conv2DLayer(net['s1_conv3_1'], 256, 3, pad=1, W=GlorotUniform('relu')))  
        net['s1_pool3'] = lasagne.layers.Pool2DLayer(net['s1_conv3_2'], 2)
        
        net['s1_conv4_1'] = batch_norm(Conv2DLayer(net['s1_pool3'], 512, 3, pad=1, W=GlorotUniform('relu')))
        net['s1_conv4_2'] = batch_norm (Conv2DLayer(net['s1_conv4_1'], 512, 3, pad=1, W=GlorotUniform('relu')))  
        net['s1_pool4'] = lasagne.layers.Pool2DLayer(net['s1_conv4_2'], 2)
                      
        net['s1_fc1_dropout'] = lasagne.layers.DropoutLayer(net['s1_pool4'], p=0.5)
        net['s1_fc1'] = batch_norm(lasagne.layers.DenseLayer(net['s1_fc1_dropout'], num_units=256, W=GlorotUniform('relu')))

        net['s1_output'] = lasagne.layers.DenseLayer(net['s1_fc1'], num_units=136, nonlinearity=None)
        net['s1_landmarks'] = LandmarkInitLayer(net['s1_output'], self.initLandmarks)

        if self.confidenceLayer:
            net['s1_confidence'] = lasagne.layers.DenseLayer(net['s1_fc1'], num_units=2, W=GlorotUniform('relu'), nonlinearity=lasagne.nonlinearities.softmax)

        for i in range(1, self.nStages):
            self.addDANStage(i + 1, net)

        net['output'] = net['s' + str(self.nStages) + '_landmarks']
        if self.confidenceLayer:
            net['output'] = lasagne.layers.ConcatLayer([net['output'], net['s1_confidence']])

        return net

    def loadNetwork(self, filename):
        print('Loading network...')

        with np.load(filename) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files) - 5)]
            self.errors = f["errors"].tolist()
            self.errorsTrain = f["errorsTrain"].tolist()
            self.meanImg = f["meanImg"]
            self.stdDevImg = f["stdDevImg"]
            self.initLandmarks = f["initLandmarks"]
                
        self.initializeNetwork()
        nParams = len(lasagne.layers.get_all_param_values(self.network))
        lasagne.layers.set_all_param_values(self.network, param_values[:nParams])
        
    def processImg(self, img, inputLandmarks):
        inputImg, transform = self.CropResizeRotate(img, inputLandmarks)
        inputImg = inputImg - self.meanImg
        inputImg = inputImg / self.stdDevImg

        output = self.generate_network_output([inputImg])[0][0]
        if self.confidenceLayer:
            landmarkOutput = output[:-2]
            confidenceOutput = output[-2:]

            landmarks = landmarkOutput.reshape((-1, 2))
            confidence = confidenceOutput[1]

            return np.dot(landmarks - transform[1], np.linalg.inv(transform[0])), confidence
        else:
            landmarks = output.reshape((-1, 2))
            return np.dot(landmarks - transform[1], np.linalg.inv(transform[0]))

    def processNormalizedImg(self, img):
        inputImg = img.astype(np.float32)
        inputImg = inputImg - self.meanImg
        inputImg = inputImg / self.stdDevImg

        output = self.generate_network_output([inputImg])[0][0]
        if self.confidenceLayer:
            landmarkOutput = output[:-2]
            confidenceOutput = output[-2:]

            landmarks = landmarkOutput.reshape((-1, 2))
            confidence = confidenceOutput[1]
            return landmarks, confidence
        else:
            landmarks = output.reshape((-1, 2))       
            return landmarks

    def CropResizeRotate(self, img, inputShape):
        A, t = utils.bestFit(self.initLandmarks, inputShape, True)
    
        A2 = np.linalg.inv(A)
        t2 = np.dot(-t, A2)

        outImg = np.zeros((self.nChannels, self.imageHeight, self.imageWidth), dtype=np.float32)
        for i in range(img.shape[0]):
            outImg[i] = ndimage.interpolation.affine_transform(img[i], A2, t2[[1, 0]], output_shape=(self.imageHeight, self.imageWidth))

        return outImg, [A, t]
