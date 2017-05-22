from lasagne.layers import Layer

class LandmarkInitLayer(Layer):
    def __init__(self, increments, init_landmarks, **kwargs):
        super(LandmarkInitLayer, self).__init__(increments, **kwargs)

        self.init_landmarks = init_landmarks.flatten()

    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_output_for(self, input, **kwargs):      
        output = input + self.init_landmarks

        return output
