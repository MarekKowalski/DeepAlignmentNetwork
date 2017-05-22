from __future__ import print_function

import os
import time
import datetime
import lasagne
from lasagne.layers import Conv2DLayer, batch_norm
from lasagne.init import GlorotUniform

from matplotlib import pyplot as plt
import numpy as np
import theano
import theano.tensor as T

from AffineTransformLayer import AffineTransformLayer
from TransformParamsLayer import TransformParamsLayer
from LandmarkImageLayer import LandmarkImageLayer
from LandmarkInitLayer import LandmarkInitLayer
from LandmarkTranformLayer import LandmarkTransformLayer

class FaceAlignmentTraining(object):
    def __init__(self, nStages, stagesToTrain):
        self.batchsize = 64
        self.landmarkPatchSize = 16

        self.data = theano.tensor.tensor4('inputs', dtype=theano.config.floatX)
        self.targets = theano.tensor.tensor4('targets')

        self.errors = []
        self.errorsTrain = []

        self.nStages = nStages
        self.stagesToTrain = stagesToTrain

    def initializeNetwork(self):
        self.layers = self.createCNN()
        self.network = self.layers['output']
        
        self.prediction = lasagne.layers.get_output(self.network, deterministic=False)
        self.prediction_test = lasagne.layers.get_output(self.network, deterministic=True)


        self.generate_network_output = theano.function([self.data], [self.prediction])
        self.generate_network_output_deterministic = theano.function([self.data], [self.prediction_test])
        
        self.loss = self.landmarkErrorNorm(self.prediction, self.targets)
        self.test_loss = self.landmarkErrorNorm(self.prediction_test, self.targets)

        self.test_fn = theano.function([self.data, self.targets], self.test_loss)  

    def landmarkPairErrorNorm(self, output, landmarks):
        gtLandmarks = landmarks[1]
        initLandmarks = landmarks[0]

        transformedLandmarks = T.reshape(output[:136], (68, 2))
      
        meanError = T.mean(T.sqrt(T.sum((transformedLandmarks - gtLandmarks)**2, axis=1)))
        eyeDist = (T.mean(gtLandmarks[36:42], axis=0) - T.mean(gtLandmarks[42:48], axis=0)).norm(2)
        res = meanError / eyeDist

        return res

    def landmarkErrorNorm(self, transforms, landmarks):
        errors, updates = theano.scan(self.landmarkPairErrorNorm, [transforms, landmarks])
            
        return T.mean(errors)

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

        for i in range(1, self.nStages):
            self.addDANStage(i + 1, net)

        net['output'] = net['s' + str(self.nStages) + '_landmarks']

        return net

    def getLabelsForDataset(self, imageServer):
        nSamples = imageServer.gtLandmarks.shape[0]
        nLandmarks = imageServer.gtLandmarks.shape[1]

        y = np.zeros((nSamples, 2, nLandmarks, 2), dtype=np.float32)
        y[:, 0] = imageServer.initLandmarks
        y[:, 1] = imageServer.gtLandmarks

        return y

    def loadData(self, trainSet, validationSet):
        self.nSamples = trainSet.gtLandmarks.shape[0]
        self.imageHeight = trainSet.imgSize[0]
        self.imageWidth = trainSet.imgSize[1]
        self.nChannels = trainSet.imgs.shape[1]

        self.Xtrain = trainSet.imgs
        self.Xvalid = validationSet.imgs

        self.Ytrain = self.getLabelsForDataset(trainSet)
        self.Yvalid = self.getLabelsForDataset(validationSet)

        self.testIdxsTrainSet = range(len(self.Xvalid))
        self.testIdxsValidSet = range(len(self.Xvalid))

        self.meanImg = trainSet.meanImg
        self.stdDevImg = trainSet.stdDevImg
        self.initLandmarks = trainSet.initLandmarks[0]

            
    def iterate_minibatches(self, inputs, targets, batchsize, shuffle=False):
        assert len(inputs) == len(targets)
        if shuffle:
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)
        for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            yield inputs[excerpt], targets[excerpt]

    def loadNetwork(self, filename):
        print('Loading network...')

        with np.load(filename) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files) - 5)]
            self.errors = f["errors"].tolist()
            self.errorsTrain = f["errorsTrain"].tolist()
            self.meanImg = f["meanImg"]
            self.stdDevImg = f["stdDevImg"]
            self.initLandmarks = f["initLandmarks"]

        prev_parameters = lasagne.layers.get_all_param_values(self.network);
        
        if (len(prev_parameters)!=len(param_values)):
            print("Loading warning: different network shape, trying to do something useful")
    
        n_assigned_parameters = 0
        for i in range(len(prev_parameters)):
            if prev_parameters[i].shape == param_values[n_assigned_parameters].shape:
                prev_parameters[i] = param_values[n_assigned_parameters]
                n_assigned_parameters += 1
                if n_assigned_parameters == len(param_values):
                    break
            else:
                break
                
        if (n_assigned_parameters != len(param_values)):
            print("Assigned " + str(n_assigned_parameters) + "/" + str(len(param_values)) + " parameters")

        param_values = prev_parameters 

        lasagne.layers.set_all_param_values(self.network, param_values)

    def saveNetwork(self, dir="../networks/"):
        if not os.path.exists(dir):
            os.makedirs(dir)
        network_filename = dir + "network_" + str(len(self.errors)).zfill(5) + "_" + str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"))

        np.savez(network_filename, *lasagne.layers.get_all_param_values(self.layers["output"]), errors = self.errors, errorsTrain = self.errorsTrain, 
                 meanImg = self.meanImg, stdDevImg=self.stdDevImg, initLandmarks=self.initLandmarks)        

    def getOutput(self, imgs, chunkSize=50):
        output = []
        nImages = len(imgs)
        nChunks = 1 + nImages / chunkSize

        imgs = np.array_split(imgs, nChunks)
        for imgSet in imgs:
            if len(output) == 0:
                output = self.generate_network_output_deterministic(imgSet)[0]
            else:
                output = np.vstack((output, self.generate_network_output_deterministic(imgSet)[0]))

        return output

    def validateNetwork(self):        
        error = self.getErrors(self.Xvalid, self.Yvalid, self.test_fn, self.testIdxsValidSet)
        errorTrain = self.getErrors(self.Xtrain, self.Ytrain, self.test_fn, self.testIdxsTrainSet)
        print("Validation error: " + str(error))
        print("Train error: " + str(errorTrain))
        self.errors.append(error)
        self.errorsTrain.append(errorTrain)
        self.drawErrors()

        textRepresentation = np.column_stack((range(len(self.errors)), self.errors, self.errorsTrain))
        
        np.savetxt("../errors.txt", textRepresentation)          

    def drawErrors(self):
        plt.plot(self.errors)
        plt.plot(self.errorsTrain)
        plt.ylim(ymax=np.max([self.errors[0], self.errorsTrain[0]]))
        plt.savefig("../errors.jpg")
        plt.clf()

    def getErrors(self, X, y, loss, idxs, chunkSize=50):
        error = 0

        nImages = len(idxs)
        nChunks = 1 + nImages / chunkSize

        idxs = np.array_split(idxs, nChunks)
        for i in range(len(idxs)):
            error += loss(X[idxs[i]], y[idxs[i]])

        error = error / len(idxs)
        return error

    def getParamsForStage(self, stageIdx):        
        if stageIdx == 0:
            return lasagne.layers.get_all_params(self.layers['s1_landmarks'], trainable=True)

        allParams = lasagne.layers.get_all_params(self.layers['s' + str(stageIdx + 1) + '_landmarks'], trainable=True)
        prevParams = lasagne.layers.get_all_params(self.layers['s' + str(stageIdx) + '_landmarks'], trainable=True)

        params = [x for x in allParams if x not in prevParams]

        return params

    def train(self, learning_rate = 0.05, num_epochs=10000):       
        params = []
        for stage in self.stagesToTrain:
            params += self.getParamsForStage(stage) 

        updates = lasagne.updates.adam(self.loss, params, learning_rate=learning_rate)     

        self.train_fn = theano.function([self.data, self.targets], self.loss, updates=updates)  
       

        print("Starting training...")
        self.validateNetwork()        
        lowestError = np.min(self.errors)

        for epoch in range(num_epochs):
            print("Starting epoch " + str(epoch))

            train_err = 0
            train_batches = 0
            start_time = time.time()

            for batch in self.iterate_minibatches(self.Xtrain, self.Ytrain, self.batchsize, True):                
                inputs, targets = batch
                train_batches += 1
                train_err += self.train_fn(inputs, targets)
                
                if train_batches %40 == 0:
                    self.validateNetwork()
                    if self.errors[-1] < lowestError:
                        self.saveNetwork("../")
                        lowestError = self.errors[-1]

            print(train_batches)

            self.validateNetwork()
            print("training loss:\t\t{:.6f}".format(train_err / train_batches))
            self.saveNetwork()
