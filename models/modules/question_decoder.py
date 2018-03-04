import tensorflow as tf
import numpy as np

class question_decoder():
	"""
		Takes in previous state, and returns the question for that time step
		###Dimensions
		Previous state: (batch_size, hidden_dimension)
		Question: (batch_size, question_length, indices)
	"""
	def __init__(self, hidden_dimension, start_token, max_question_length):
		self.hidden_dimension = hidden_dimension
		self.start_token = start_token
		self.max_question_length = max_question_length

	def build_graph(self, states, scope):

		with tf.varible_scope(scope):
			start_tokens = tf.tile(self.start_token, [tf.shape(states)[0],1])
			helper = tf.contrib.seq2seq.TrainingHelper(start_tokens,sequence_length,scope)
			cells = [tf.contrib.BasicRNNCell(self.hidden_dimension), tf.contrib.BasicLSTMCell(self.hidden_dimension)]
			cell = tf.contrib.rnn.MultiRNNCell(cells)
			decoder = tf.contrib.seq2seq.BasicDecoder(cell,helper,states,output_layer=tf.contrib.layers.softmax)
			final_outputs, final_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder=decoder, 
														impute_finished=True, maximum_iterations=self.max_question_length)
			return final_outputs, final_state, final_sequence_lengths