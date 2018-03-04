import tensorflow as tf
import numpy as np

class question_decoder():
	"""
		Takes in previous state, and returns the question for that time step
		###Dimensions
		Previous state: (batch_size, hidden_dimension)
		Question: (batch_size, question_length, indices)
	"""
	def __init__(self, hidden_dimension, start_token, end_token, max_question_length, scope):
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
		self.end_token = end_token
		self.max_question_length = max_question_length
		self.scope = scope
		self.add_cells()

	def add_cells(self):
		"""
		Builds the graph to create the RNN's which do the dirty work
		TODO: Differentiate between supervised pre-training and RL-training:
		===================================
		"""
		with tf.varible_scope(self.scope):

			cells = [tf.contrib.BasicRNNCell(self.hidden_dimension), tf.contrib.BasicLSTMCell(self.hidden_dimension)]
			#stacked cell
			self.cell = tf.contrib.rnn.MultiRNNCell(cells)

	def generate_question(self, states, true_questions, true_question_lengths, flag=True, embedding = None):
		"""
		Builds the graph to take in the state, and generate a question
		TODO: Differentiate between supervised pre-training and RL-training:
		===================================
		INPUTS:
		states: float of shape (batch_size, hidden_dimension) - The state/history encoding for this round of dialog
		true_questions: float of shape (batch_size, max_question_length, embedding_size) || Assumed that questions have been padded to max_question_size
		true_question_lengths: int of shape(batch_size) - How long is the actual question?
		flag: bool True: supervised pretraining|| False: RL training
		embedding: embedding matrix of size (embedding_size, vocabulary_size)
		===================================
		OUTPUTS:
		final_outputs: float of shape (batch_size, max_sequence_length, hidden_size): The sequence of output vectors for every timestep
		final_state = (batch_size, hidden_size): The final hidden state for every question in the batch
		final_sequence_lengths = (batch_size): The actual length of the questions
		"""
		with tf.varible_scope(self.scope):
			#start_tokens to 
			# start_tokens = tf.tile(self.start_token, [tf.shape(states)[0],1])
			if flag:
				helper = tf.contrib.seq2seq.TrainingHelper(true_questions, true_question_lengths, self.scope)
			else:
				start_tokens = tf.tile(self.start_token, [tf.shape(states)[0]])
				helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding = embedding, start_tokens = start_tokens, end_token = self.end_token)
			#decoder instance
			decoder = tf.contrib.seq2seq.BasicDecoder(self.cell, helper, states)
			#final sequence of outputs
			#final_outputs = (batch_size, max_sequence_length, hidden_size)
			#final_state = (batch_size, hidden_size)
			#final_sequence_lengths = (batch_size)
			final_outputs, final_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder=decoder, 
														impute_finished=True, maximum_iterations=self.max_question_length)
		
			return final_outputs, final_state, final_sequence_lengths
