import tensorflow as tf

class FeatureRegressor(object):
    def __init__(self, image_dimension, scope):
        """
        Initialization function
        ====================
        INPUTS:
        image_dimension: int - shape of the image respresentation
        ====================
        """
        self.image_dimension = image_dimension
        self.scope = scope

    def generate_image_prediction(self, state):
        """
        Builds the graph to take in the question, answer and generates the new fact
        ===================================
        INPUTS:
        state: float of shape (batch_size, hidden_dimensio ) - The state-history at this round of dialog
        ===================================
        OUTPUTS:
        image_prediction: float of shape (batch_size, image_prediction) - The new image prediction based on dialog so far
        """
        ##Assumed that fully connected layer has no activation at output!
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            image_prediction = tf.contrib.layers.fully_connected(state, self.image_dimension, activation_fn = None)
            return image_prediction
