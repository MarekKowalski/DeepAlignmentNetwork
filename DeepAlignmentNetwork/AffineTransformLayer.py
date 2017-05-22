from lasagne.layers import MergeLayer
import theano
from theano import tensor as T
import numpy as np


class AffineTransformLayer(MergeLayer):
    def __init__(self, images, transform_params, output_shape=None, **kwargs):
        super(AffineTransformLayer, self).__init__([images, transform_params], **kwargs)

        img_shape, _ = self.input_shapes
        
        if output_shape is None:
            self.out_img_height = img_shape[2]
            self.out_img_width = img_shape[3]
        else:
            self.out_img_height = output_shape[0]
            self.out_img_width = output_shape[1]

        self.in_img_height = img_shape[2]
        self.in_img_width = img_shape[3]
        
    def affine_transform(self, img, A, t):
        pixels = [(x, y) for x in range(self.out_img_width) for y in range(self.out_img_height)]
        pixels = np.array(pixels, dtype=np.float32)

        outPixels = T.dot(pixels, A) + t

        outPixels = T.set_subtensor(outPixels[:, 0], T.clip(outPixels[:, 0], 0, self.in_img_height - 2))
        outPixels = T.set_subtensor(outPixels[:, 1], T.clip(outPixels[:, 1], 0, self.in_img_height - 2))

        outPixelsMinMin = outPixels.astype('int32')
        outPixelsMaxMin = outPixelsMinMin + [1, 0]
        outPixelsMinMax = outPixelsMinMin + [0, 1]
        outPixelsMaxMax = outPixelsMinMin + [1, 1]

        dx = outPixels[:, 0] - outPixelsMinMin[:, 0]
        dy = outPixels[:, 1] - outPixelsMinMin[:, 1]

        pixels = pixels.astype('int32')
    
        outImg = T.zeros((1, self.out_img_height, self.out_img_width))
        outImg = T.inc_subtensor(outImg[0, pixels[:, 1], pixels[:, 0]], (1 - dx) * (1 - dy) * img[outPixelsMinMin[:, 1], outPixelsMinMin[:, 0]])
        outImg = T.inc_subtensor(outImg[0, pixels[:, 1], pixels[:, 0]], dx * (1 - dy) * img[outPixelsMaxMin[:, 1], outPixelsMaxMin[:, 0]])
        outImg = T.inc_subtensor(outImg[0, pixels[:, 1], pixels[:, 0]], (1 - dx) * dy * img[outPixelsMinMax[:, 1], outPixelsMinMax[:, 0]])
        outImg = T.inc_subtensor(outImg[0, pixels[:, 1], pixels[:, 0]], dx * dy * img[outPixelsMaxMax[:, 1], outPixelsMaxMax[:, 0]])

        return outImg

    def affine_transform_helper(self, img, transform):
        A = T.zeros((2, 2))     
        
        A = T.set_subtensor(A[0, 0], transform[0])
        A = T.set_subtensor(A[0, 1], transform[1])
        A = T.set_subtensor(A[1, 0], transform[2])
        A = T.set_subtensor(A[1, 1], transform[3])
        t = transform[4:6]

        A = T.nlinalg.matrix_inverse(A)
        t = T.dot(-t, A)

        return self.affine_transform(img[0], A, t)

    def get_output_shape_for(self, input_shapes):
        return (None, 1, self.out_img_height, self.out_img_width)

    def get_output_for(self, inputs, **kwargs):      
        outImgs, updates = theano.scan(self.affine_transform_helper, inputs)

        return outImgs
