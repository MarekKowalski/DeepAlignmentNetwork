from lasagne.layers import Layer
import theano
from theano import tensor as T

class TransformParamsLayer(Layer):
    def __init__(self, shape_updates, mean_shape, **kwargs):
        super(TransformParamsLayer, self).__init__(shape_updates, **kwargs)

        self.mean_shape = mean_shape

    def get_output_shape_for(self, input_shapes):
        return (None, 6)

    def bestFit(self, transformed_shape):
        destination = self.mean_shape
        source = transformed_shape.reshape((-1, 2))

        destMean = T.mean(destination, axis=0)
        srcMean = T.mean(source, axis=0)

        srcVec = (source - srcMean).flatten()
        destVec = (destination - destMean).flatten()

        a = T.dot(srcVec, destVec) / T.nlinalg.norm(srcVec, 2)**2
        b = 0
        for i in range(self.mean_shape.shape[0]):
            b += srcVec[2*i] * destVec[2*i+1] - srcVec[2*i+1] * destVec[2*i] 
        b = b / T.nlinalg.norm(srcVec, 2)**2        
    
        A = T.zeros((2, 2))
        A = T.set_subtensor(A[0, 0], a)
        A = T.set_subtensor(A[0, 1], b)
        A = T.set_subtensor(A[1, 0], -b)
        A = T.set_subtensor(A[1, 1], a)
        srcMean = T.dot(srcMean, A)        
        
        return T.concatenate((A.flatten(), destMean - srcMean))

    def get_output_for(self, input, **kwargs):      
        transforms, updates = theano.scan(self.bestFit, [input])

        return transforms