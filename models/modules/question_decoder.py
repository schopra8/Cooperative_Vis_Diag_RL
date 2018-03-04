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
		"""
		Initialization function
		====================
		INPUTS:
		hidden_dimension: int - shape of the hidden state for the LSTM/RNN cells used
		start_token : float of shape (embedding_size) - word embedding of the start_token
		max_question_length : int - length of longest question
		====================
		"""
		self.hidden_dimension = hidden_dimension
		self.start_token = start_token
		self.max_question_length = max_question_length

	def build_graph(self, states, questions, question_lengths, scope):
		"""
		Builds the graph to take in the state, and generate a question
		TODO: Differentiate between supervised pre-training and RL-training:
		===================================
		INPUTS:
		states: float of shape (batch_size, hidden_dimension) - The state/history encoding for this round of dialog
		questions: float of shape (batch_size, max_question_length, embedding_size) || Assumed that questions have been padded to max_question_size
		question_lengths: int of shape(batch_size) - How long is the actual question?
		scope: Scope for all variables created in this call to build_graph()
		"""
		
		with tf.varible_scope(scope):
			#start_tokens to 
			# start_tokens = tf.tile(self.start_token, [tf.shape(states)[0],1])
			helper = tf.contrib.seq2seq.TrainingHelper(questions, question_lengths,scope)
			cells = [tf.contrib.BasicRNNCell(self.hidden_dimension), tf.contrib.BasicLSTMCell(self.hidden_dimension)]
			#stacked cell
			cell = tf.contrib.rnn.MultiRNNCell(cells)
			#decoder instance
			decoder = tf.contrib.seq2seq.BasicDecoder(cell helper, states)
			#final sequence of outputs
			#final_outputs = (batch_size, max_sequence_length, hidden_size)
			#final_state = (batch_size, hidden_size)
			#final_sequence_lengths = (batch_size)
			final_outputs, final_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder=decoder, 
														impute_finished=True, maximum_iterations=self.max_question_length)
			
			return final_outputs, final_state, final_sequence_lengths