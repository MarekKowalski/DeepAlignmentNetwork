from lasagne.layers import Layer
import theano
from theano import tensor as T
import itertools
import numpy as np

class LandmarkImageLayer(Layer):
    def __init__(self, increments, img_shape, patch_size, **kwargs):
        super(LandmarkImageLayer, self).__init__(increments, **kwargs)

        self.img_shape = img_shape
        self.patch_size = patch_size
        self.half_size = patch_size / 2

        self.offsets = np.array(list(itertools.product(range(-self.half_size, self.half_size + 1), range(-self.half_size, self.half_size + 1))))

    def get_output_shape_for(self, input_shape):        
        return (input_shape[0], 1, self.img_shape[0], self.img_shape[1])

    def draw_landmarks_helper(self, landmark):
        img = T.zeros((1, self.img_shape[0], self.img_shape[1]))
        
        intLandmark = landmark.astype('int32')
        locations = self.offsets + intLandmark
        dxdy = landmark - intLandmark

        offsetsSubPix = self.offsets - dxdy
        vals = 1 / (1 + T.sqrt(T.sum(offsetsSubPix * offsetsSubPix, axis=1) + 1e-6))

        img = T.set_subtensor(img[0, locations[:, 1], locations[:, 0]], vals)
        return img

    def draw_landmarks(self, input):      
        landmarks = input.reshape((-1, 2))
        landmarks = T.set_subtensor(landmarks[:, 0], T.clip(landmarks[:, 0], self.half_size, self.img_shape[1] - 1 - self.half_size))
        landmarks = T.set_subtensor(landmarks[:, 1], T.clip(landmarks[:, 1], self.half_size, self.img_shape[0] - 1 - self.half_size))

        imgs, updates = theano.scan(self.draw_landmarks_helper, landmarks)        
        img = T.max(imgs, 0)

        return img

    def get_output_for(self, input, **kwargs):      
        output, updates = theano.scan(self.draw_landmarks, [input])

        return output

